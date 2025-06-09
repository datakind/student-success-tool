import pytest
from unittest.mock import Mock
import mlflow
from student_success_tool.modeling.registration import register_mlflow_model


@pytest.fixture
def mock_client():
    client = Mock(spec=mlflow.tracking.MlflowClient)
    return client


def test_registers_new_model_and_sets_tag(mock_client):
    run_id = "abc123"
    model_name = "my_model"
    institution_id = "inst"
    catalog = "main"
    version = Mock()
    version.version = 5

    # Simulate: model doesn't exist → create it
    mock_client.create_registered_model.side_effect = None
    mock_client.get_run.return_value.data.tags = {}

    mock_client.create_model_version.return_value = version

    register_mlflow_model(
        model_name=model_name,
        institution_id=institution_id,
        run_id=run_id,
        catalog=catalog,
        mlflow_client=mock_client,
    )

    model_path = f"{catalog}.{institution_id}_gold.{model_name}"
    model_uri = f"runs:/{run_id}/model"

    mock_client.create_registered_model.assert_called_once_with(name=model_path)
    mock_client.get_run.assert_called_once_with(run_id)
    mock_client.create_model_version.assert_called_once_with(
        name=model_path,
        source=model_uri,
        run_id=run_id,
    )
    mock_client.set_tag.assert_called_once_with(run_id, "model_registered", "true")
    mock_client.set_registered_model_alias.assert_called_once_with(
        model_path, "Staging", version.version
    )


def test_skips_if_tag_indicates_already_registered(mock_client):
    mock_client.get_run.return_value.data.tags = {"model_registered": "true"}

    register_mlflow_model(
        model_name="my_model",
        institution_id="inst",
        run_id="abc123",
        catalog="main",
        mlflow_client=mock_client,
    )

    mock_client.create_model_version.assert_not_called()
    mock_client.set_tag.assert_not_called()
    mock_client.set_registered_model_alias.assert_not_called()


def test_handles_existing_registered_model_gracefully(mock_client):
    mock_client.create_registered_model.side_effect = mlflow.exceptions.MlflowException(
        "RESOURCE_ALREADY_EXISTS"
    )
    mock_client.get_run.return_value.data.tags = {}
    mock_client.create_model_version.return_value.version = 1

    register_mlflow_model(
        model_name="m",
        institution_id="inst",
        run_id="run1",
        catalog="main",
        mlflow_client=mock_client,
    )

    mock_client.create_model_version.assert_called()


def test_raises_if_tag_check_fails(mock_client):
    mock_client.get_run.side_effect = mlflow.exceptions.MlflowException("Bad request")

    with pytest.raises(mlflow.exceptions.MlflowException):
        register_mlflow_model(
            model_name="m",
            institution_id="inst",
            run_id="bad_run",
            catalog="main",
            mlflow_client=mock_client,
        )


def test_skips_setting_alias_if_none(mock_client):
    mock_client.get_run.return_value.data.tags = {}
    mock_client.create_model_version.return_value.version = 1

    register_mlflow_model(
        model_name="m",
        institution_id="inst",
        run_id="run1",
        catalog="main",
        model_alias=None,
        mlflow_client=mock_client,
    )

    mock_client.set_registered_model_alias.assert_not_called()
