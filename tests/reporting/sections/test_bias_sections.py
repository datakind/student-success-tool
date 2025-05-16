import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.bias_sections import register_bias_sections
from student_success_tool.reporting.utils.formatting import Formatting


@pytest.fixture
def mock_card(tmp_path):
    card = MagicMock()
    card.run_id = "123"
    card.assets_folder = str(tmp_path)
    card.format = Formatting()
    return card


@patch("student_success_tool.reporting.sections.bias_sections.utils.download_artifact")
def test_register_bias_sections_with_data(mock_download, mock_card, tmp_path):
    # Create mock bias CSV
    bias_csv = tmp_path / "high_bias_flags.csv"
    df = pd.DataFrame(
        {
            "group": ["Gender"],
            "subgroups": ["F vs M"],
            "fnr_percentage_difference": [0.17],
            "type": ["non-overlapping confidence intervals with a p-value of 0.001"],
            "split_name": ["test"],
        }
    )
    df.to_csv(bias_csv, index=False)

    def mock_download_side_effect(run_id, local_folder, artifact_path, **kwargs):
        if artifact_path.startswith("bias_flags/"):
            return str(bias_csv)
        elif artifact_path.startswith("fnr_plots/"):
            return "<img src='plot.png'>"
        raise ValueError("Unexpected path")

    mock_download.side_effect = mock_download_side_effect

    registry = SectionRegistry()
    register_bias_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "Disparities by Student Group" in rendered["bias_summary_section"]
    assert "F" in rendered["bias_summary_section"]
    assert "17% difference" in rendered["bias_summary_section"]
    assert "non-overlapping" in rendered["bias_summary_section"]
    assert "plot.png" in rendered["bias_summary_section"]


@patch("student_success_tool.reporting.sections.bias_sections.utils.download_artifact")
def test_register_bias_sections_no_data(mock_download, mock_card):
    mock_download.side_effect = Exception("No bias data")

    registry = SectionRegistry()
    register_bias_sections(mock_card, registry)
    rendered = registry.render_all()

    assert (
        "No statistically significant disparities were found"
        in rendered["bias_summary_section"]
    )
