import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from student_success_tool.reporting.sections.registry import SectionRegistry
from student_success_tool.reporting.sections.custom import (
    evaluation_sections as custom_evaluation_sections,
)
from student_success_tool.reporting.utils.formatting import Formatting


@pytest.fixture
def mock_card_with_aliases(tmp_path):
    card = MagicMock()
    card.run_id = "456"
    card.assets_folder = str(tmp_path)
    card.format = Formatting()
    card.cfg.student_group_aliases = {
        "gender": "Gender Identity",
        "race_demo": "Race/Ethnicity",
    }
    return card


@pytest.fixture
def mock_card_without_aliases(tmp_path):
    card = MagicMock()
    card.run_id = "789"
    card.assets_folder = str(tmp_path)
    card.format = Formatting()
    card.cfg.student_group_aliases = None  # Simulate missing config
    return card


@patch(
    "student_success_tool.reporting.sections.custom.evaluation_sections.utils.list_paths_in_directory"
)
@patch(
    "student_success_tool.reporting.sections.custom.evaluation_sections.utils.download_artifact"
)
def test_register_evaluation_sections_with_aliases(
    mock_download, mock_list_paths, mock_card_with_aliases, tmp_path
):
    # Prepare dummy CSV
    csv_path = tmp_path / "metrics.csv"
    df = pd.DataFrame({"Metric": ["Accuracy", "Recall"], "Value": [0.95, 0.88]})
    df.to_csv(csv_path, index=False)

    mock_list_paths.return_value = [
        "group_metrics/bias_test_gender_metrics.csv",
        "group_metrics/perf_test_gender_metrics.csv",
        "group_metrics/bias_test_race_demo_metrics.csv",
        "group_metrics/perf_test_race_demo_metrics.csv",
    ]
    mock_download.return_value = str(csv_path)

    registry = SectionRegistry()
    custom_evaluation_sections.register_evaluation_sections(
        mock_card_with_aliases, registry
    )
    rendered = registry.render_all()

    assert "Gender Identity Bias Metrics" in rendered["evaluation_by_group_section"]
    assert "Race/Ethnicity Bias Metrics" in rendered["evaluation_by_group_section"]
    assert "| Accuracy | 0.95 |" in rendered["evaluation_by_group_section"]


@patch(
    "student_success_tool.reporting.sections.custom.evaluation_sections.utils.list_paths_in_directory"
)
@patch(
    "student_success_tool.reporting.sections.custom.evaluation_sections.utils.download_artifact"
)
def test_register_evaluation_sections_fallback_to_friendly_case(
    mock_download, mock_list_paths, mock_card_without_aliases, tmp_path
):
    csv_path = tmp_path / "metrics.csv"
    df = pd.DataFrame({"Metric": ["Precision", "F1"], "Value": [0.81, 0.78]})
    df.to_csv(csv_path, index=False)

    mock_list_paths.return_value = [
        "group_metrics/bias_test_race_demo_metrics.csv",
        "group_metrics/perf_test_race_demo_metrics.csv",
    ]
    mock_download.return_value = str(csv_path)

    registry = SectionRegistry()
    custom_evaluation_sections.register_evaluation_sections(
        mock_card_without_aliases, registry
    )
    rendered = registry.render_all()

    assert "Race Demo Bias Metrics" in rendered["evaluation_by_group_section"]
    assert "| Precision | 0.81 |" in rendered["evaluation_by_group_section"]


@patch(
    "student_success_tool.reporting.sections.custom.evaluation_sections.utils.list_paths_in_directory"
)
@patch(
    "student_success_tool.reporting.sections.custom.evaluation_sections.utils.download_artifact"
)
def test_register_evaluation_sections_handles_download_failure(
    mock_download, mock_list_paths, mock_card_with_aliases
):
    mock_list_paths.return_value = [
        "group_metrics/bias_test_gender_metrics.csv",
        "group_metrics/perf_test_gender_metrics.csv",
    ]
    mock_download.side_effect = Exception("Failed to download")

    registry = SectionRegistry()
    custom_evaluation_sections.register_evaluation_sections(
        mock_card_with_aliases, registry
    )
    rendered = registry.render_all()

    assert "Could not load data." in rendered["evaluation_by_group_section"]
