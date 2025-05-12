from unittest.mock import patch, MagicMock
from student_success_tool.reporting.utils import utils


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


def test_embed_image_relative_path(tmp_path):
    test_file = tmp_path / "example.png"
    test_file.write_text("image content")
    result = utils.embed_image("Test Image", test_file, max_width_pct=50, alignment="left")
    assert "img src=" in result
    assert 'max-width: 50%' in result
    assert 'alt="Test Image"' in result
    assert "display: block; margin-left: 0;" in result


@patch("student_success_tool.reporting.utils.utils.mlflow.artifacts.list_artifacts")
def test_list_paths_in_directory(mock_list):
    mock_list.return_value = [MagicMock(path="group_metrics/test_gender_metrics.csv")]
    result = utils.list_paths_in_directory(run_id="abc123", directory="group_metrics")
    assert result == ["group_metrics/test_gender_metrics.csv"]
