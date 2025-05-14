import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from student_success_tool.reporting.model_card.base import ModelCard


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.model = MagicMock(
        mlflow_model_uri="uri", framework="sklearn", run_id="123", experiment_id="456"
    )
    cfg.institution_id = "inst"
    cfg.institution_name = "TestInstitution"
    cfg.modeling.feature_selection.collinear_threshold = 0.9
    cfg.modeling.feature_selection.low_variance_threshold = 0.01
    cfg.modeling.feature_selection.incomplete_threshold = 0.05
    cfg.split_col = None
    return cfg


def test_init_defaults(mock_config):
    card = ModelCard(config=mock_config, catalog="catalog", model_name="inst_my_model")
    assert card.model_name == "inst_my_model"
    assert card.uc_model_name == "catalog.inst_gold.inst_my_model"
    assert card.assets_folder == "card_assets"
    assert isinstance(card.context, dict)


@patch("student_success_tool.reporting.model_card.dataio.models.load_mlflow_model")
def test_load_model_success(mock_load_model, mock_config):
    card = ModelCard(config=mock_config, catalog="catalog", model_name="inst_my_model")
    card.load_model()
    mock_load_model.assert_called_once_with("uri", "sklearn")
    assert card.run_id == "123"
    assert card.experiment_id == "456"


def test_find_model_version_found(mock_config):
    card = ModelCard(config=mock_config, catalog="catalog", model_name="inst_my_model")
    card.run_id = "123"
    mock_version = MagicMock(run_id="123", version="5")
    card.client.search_model_versions = MagicMock(return_value=[mock_version])
    card.find_model_version()
    assert card.context["version_number"] == "5"


def test_find_model_version_not_found(mock_config):
    card = ModelCard(config=mock_config, catalog="catalog", model_name="inst_my_model")
    card.run_id = "999"
    card.client.search_model_versions = MagicMock(return_value=[])
    card.find_model_version()
    assert card.context["version_number"] == "Unknown"


def test_get_feature_metadata_success(mock_config):
    card = ModelCard(config=mock_config, catalog="catalog", model_name="inst_my_model")
    card.model = MagicMock()
    card.model.named_steps = {
        "column_selector": MagicMock(get_params=lambda: {"cols": ["a", "b", "c"]})
    }
    metadata = card.get_feature_metadata()
    assert metadata["number_of_features"] == "3"
    assert metadata["collinearity_threshold"] == "0.9"


@patch("student_success_tool.reporting.model_card.utils.download_static_asset")
@patch("student_success_tool.reporting.model_card.datetime")
def test_get_basic_context(mock_datetime, mock_download, mock_config):
    mock_download.return_value = "<img>Logo</img>"
    mock_datetime.now.return_value.year = 2025
    card = ModelCard(config=mock_config, catalog="catalog", model_name="inst_my_model")
    result = card.get_basic_context()
    assert result["institution_name"] == "TestInstitution"
    assert result["current_year"] == "2025"
    assert "logo" in result


def test_build_calls_all_steps(mock_config):
    card = ModelCard(config=mock_config, catalog="catalog", model_name="inst_my_model")
    for method in [
        "load_model",
        "find_model_version",
        "extract_training_data",
        "_register_sections",
        "collect_metadata",
        "render",
    ]:
        setattr(card, method, MagicMock())

    card.build()

    for method in [
        card.load_model,
        card.find_model_version,
        card.extract_training_data,
        card._register_sections,
        card.collect_metadata,
        card.render,
    ]:
        method.assert_called_once()


@patch("student_success_tool.reporting.model_card.mlflow.search_runs")
@patch(
    "student_success_tool.reporting.model_card.modeling.evaluation.extract_training_data_from_model"
)
@patch("student_success_tool.reporting.model_card.dataio.models.load_mlflow_model")
def test_extract_training_data_with_split_call_load_model(
    mock_load_model, mock_extract_data, mock_search_runs, mock_config
):
    mock_config.split_col = "split"
    df = pd.DataFrame(
        {"feature": [1, 2, 3, 4], "split": ["train", "test", "train", "val"]}
    )

    mock_extract_data.return_value = df
    mock_search_runs.return_value = pd.DataFrame({"run_id": ["123", "987"]})

    card = ModelCard(config=mock_config, catalog="catalog", model_name="inst_my_model")
    card.load_model()
    card.extract_training_data()

    assert card.context["training_dataset_size"] == 2
    assert card.context["num_runs_in_experiment"] == 2


@patch("builtins.open", new_callable=MagicMock)
def test_render_template_and_output(mock_open, mock_config):
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    mock_file.read.return_value = "Model: {institution_name}"

    card = ModelCard(config=mock_config, catalog="catalog", model_name="inst_my_model")
    card.template_path = "template.md"
    card.output_path = "output.md"
    card.context = {"institution_name": "TestInstitution"}
    card.render()

    mock_open.assert_any_call("template.md", "r")
    mock_open.assert_any_call("output.md", "w")
    mock_file.write.assert_called_once_with("Model: TestInstitution")
