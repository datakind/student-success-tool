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


@patch(
    "student_success_tool.reporting.sections.evaluation_sections.utils.list_paths_in_directory"
)
@patch(
    "student_success_tool.reporting.sections.evaluation_sections.utils.download_artifact"
)
def test_register_evaluation_sections_success(
    mock_download, mock_list_paths, mock_card, tmp_path
):
    # Setup mock CSV file
    csv_path = tmp_path / "test_file.csv"
    df = pd.DataFrame({"Metric": ["Accuracy", "Recall"], "Value": [0.9, 0.85]})
    df.to_csv(csv_path, index=False)

    mock_list_paths.return_value = [
        "group_metrics/bias_test_gender_metrics.csv",
        "group_metrics/perf_test_gender_metrics.csv",
    ]
    mock_download.return_value = str(csv_path)

    registry = SectionRegistry()
    register_evaluation_sections(mock_card, registry)
    rendered = registry.render_all()
    print(rendered["evaluation_by_group_section"])
    assert (
        "Evaluation Metrics by Student Group" in rendered["evaluation_by_group_section"]
    )
    assert "Gender Bias Metrics" in rendered["evaluation_by_group_section"]
    assert "| Accuracy | 0.9 |" in rendered["evaluation_by_group_section"]


@patch(
    "student_success_tool.reporting.sections.evaluation_sections.utils.list_paths_in_directory"
)
@patch(
    "student_success_tool.reporting.sections.evaluation_sections.utils.download_artifact"
)
def test_register_evaluation_sections_failure(
    mock_download, mock_list_paths, mock_card
):
    mock_list_paths.return_value = [
        "group_metrics/bias_test_race_metrics.csv",
        "group_metrics/perf_test_race_metrics.csv",
    ]
    mock_download.side_effect = Exception("Download error")

    registry = SectionRegistry()
    register_evaluation_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "Race Performance Metrics" in rendered["evaluation_by_group_section"]
    assert "Could not load data." in rendered["evaluation_by_group_section"]
