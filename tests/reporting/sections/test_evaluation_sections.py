import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.evaluation_sections import (
    register_evaluation_sections,
)
from student_success_tool.reporting.utils.formatting import Formatting


@pytest.fixture
def mock_card(tmp_path):
    card = MagicMock()
    card.run_id = "123"
    card.assets_folder = str(tmp_path)
    card.format = Formatting()
    return card


@patch("student_success_tool.reporting.sections.evaluation_sections.utils.list_paths_in_directory")
@patch("student_success_tool.reporting.sections.evaluation_sections.utils.download_artifact")
def test_register_evaluation_sections_success(mock_download, mock_list_paths, mock_card, tmp_path):
    # Setup two mock CSV files for bias and performance
    bias_csv_path = tmp_path / "bias_test_file.csv"
    perf_csv_path = tmp_path / "perf_test_file.csv"
    df_bias = pd.DataFrame({"Metric": ["FNR"], "Value": [0.12]})
    df_perf = pd.DataFrame({"Metric": ["Accuracy"], "Value": [0.95]})
    df_bias.to_csv(bias_csv_path, index=False)
    df_perf.to_csv(perf_csv_path, index=False)

    # Mock the artifact paths
    mock_list_paths.side_effect = [
        ["group_metrics/bias_test_gender_metrics.csv", "group_metrics/perf_test_gender_metrics.csv"],
        []  # No split artifacts in this test
    ]

    # Mock the download to return bias first, then perf
    mock_download.side_effect = [str(bias_csv_path), str(perf_csv_path)]

    registry = SectionRegistry()
    register_evaluation_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "Evaluation Metrics by Student Group" in rendered["evaluation_by_group_section"]
    assert "Gender Bias Metrics" in rendered["evaluation_by_group_section"]
    assert "Gender Performance Metrics" in rendered["evaluation_by_group_section"]
    assert "| FNR | 0.12 |" in rendered["evaluation_by_group_section"]
    assert "| Accuracy | 0.95 |" in rendered["evaluation_by_group_section"]


@patch("student_success_tool.reporting.sections.evaluation_sections.utils.list_paths_in_directory")
@patch("student_success_tool.reporting.sections.evaluation_sections.utils.download_artifact")
def test_register_evaluation_sections_failure(mock_download, mock_list_paths, mock_card):
    # Mock available bias artifact but simulate download failure
    mock_list_paths.side_effect = [
        ["group_metrics/bias_test_race_metrics.csv"],
        []  # No split artifacts in this test
    ]

    mock_download.side_effect = Exception("Download error")

    registry = SectionRegistry()
    register_evaluation_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "Race Bias Metrics" in rendered["evaluation_by_group_section"]
    assert "Could not load data." in rendered["evaluation_by_group_section"]
