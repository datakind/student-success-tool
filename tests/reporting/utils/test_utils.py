from unittest.mock import patch, MagicMock
from student_success_tool.reporting.utils import utils
from student_success_tool.reporting.utils.utils import safe_count_runs


@patch("student_success_tool.reporting.utils.utils.mlflow.artifacts.download_artifacts")
def test_download_artifact_image(mock_download):
    mock_download.return_value = "/some/folder/logo.png"
    result = utils.download_artifact(
        run_id="abc123",
        local_folder="tmp",
        artifact_path="images/logo.png",
        description="Logo",
    )
    assert "<img src=" in result
    assert 'alt="Logo"' in result


@patch("student_success_tool.reporting.utils.utils.mlflow.artifacts.download_artifacts")
def test_download_artifact_file(mock_download):
    mock_download.return_value = "/some/folder/data.csv"
    result = utils.download_artifact(
        run_id="abc123", local_folder="tmp", artifact_path="data/data.csv"
    )
    assert result == "/some/folder/data.csv"


@patch("student_success_tool.reporting.utils.utils.mlflow.artifacts.download_artifacts")
def test_download_artifact_failure_returns_none(mock_download):
    mock_download.side_effect = RuntimeError("MLflow error")
    result = utils.download_artifact(
        run_id="abc123",
        local_folder="tmp",
        artifact_path="images/logo.png",
        description="Logo",
    )
    assert result is None


def test_embed_image_relative_path(tmp_path):
    test_file = tmp_path / "example.png"
    test_file.write_text("image content")
    result = utils.embed_image(
        "Test Image", test_file, fixed_width="50mm", alignment="left"
    )
    assert "img src=" in result
    assert "width: 50mm" in result
    assert 'alt="Test Image"' in result
    assert "display: block; margin-left: 0;" in result


@patch("student_success_tool.reporting.utils.utils.mlflow.artifacts.list_artifacts")
def test_list_paths_in_directory(mock_list):
    mock_list.return_value = [MagicMock(path="group_metrics/test_gender_metrics.csv")]
    result = utils.list_paths_in_directory(run_id="abc123", directory="group_metrics")
    assert result == ["group_metrics/test_gender_metrics.csv"]


@patch("student_success_tool.reporting.utils.utils.mlflow.artifacts.list_artifacts")
def test_list_paths_in_directory_failure_returns_empty(mock_list):
    mock_list.side_effect = Exception("Something went wrong")
    result = utils.list_paths_in_directory(run_id="abc123", directory="group_metrics")
    assert result == []


@patch("student_success_tool.reporting.utils.utils.MlflowClient")
def test_safe_count_runs_success(mock_mlflow_client):
    run_page_1 = [MagicMock(), MagicMock()]
    run_page_2 = [MagicMock()]

    class RunPage(list):
        def __init__(self, items, token):
            super().__init__(items)
            self.token = token

    mock_client_instance = mock_mlflow_client.return_value
    mock_client_instance.search_runs.side_effect = [
        RunPage(run_page_1, "next-page-token"),
        RunPage(run_page_2, None),
    ]

    count = safe_count_runs("exp-123")

    assert count == 3


@patch("student_success_tool.reporting.utils.utils.MlflowClient")
def test_safe_count_runs_exception(mock_mlflow_client):
    mock_client_instance = mock_mlflow_client.return_value
    mock_client_instance.search_runs.side_effect = RuntimeError("MLflow error")

    count = safe_count_runs("exp-123")

    assert count is None
    mock_client_instance.search_runs.assert_called_once()
