import logging
import typing as t

import mlflow
import mlflow.exceptions
import mlflow.tracking

LOGGER = logging.getLogger(__name__)


def register_mlflow_model(
    model_name: str,
    institution_id: str,
    *,
    run_id: str,
    catalog: str,
    registry_uri: str = "databricks-uc",
    model_alias: t.Optional[str] = "Staging",
    mlflow_client: mlflow.tracking.MlflowClient,
) -> None:
    """
    Register an mlflow model according to one of their various recommended approaches.

    Args:
        model_name
        institution_id
        run_id
        catalog
        registry_uri
        model_alias
        mlflow_client

    References:
        - https://mlflow.org/docs/latest/model-registry.html
    """
    model_path = f"{catalog}.{institution_id}_gold.{model_name}"
    mlflow.set_registry_uri(registry_uri)

    try:
        mlflow_client.create_registered_model(name=model_path)
        LOGGER.info("new registered model '%s' successfully created", model_path)
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            LOGGER.info("model '%s' already created in registry", model_path)
        else:
            raise e

    model_uri = get_mlflow_model_uri(run_id=run_id, model_path="model")
    mv = mlflow_client.create_model_version(model_path, source=model_uri, run_id=run_id)
    if model_alias is not None:
        mlflow_client.set_registered_model_alias(
            model_path, alias=model_alias, version=mv.version
        )
    LOGGER.info("model version successfully registered at '%s'", model_path)


def get_model_name(
    *,
    institution_id: str,
    target: str,
    checkpoint: str,
    extra_info: t.Optional[str] = None,
) -> str:
    """
    Get a standard model name generated from key components, formatted as
    "{institution_id}_{target}_{checkpoint}[_{extra_info}]"
    """
    model_name = f"{institution_id}_{target}_{checkpoint}"
    if extra_info is not None:
        model_name = f"{model_name}_{extra_info}"
    return model_name


def get_mlflow_model_uri(
    *,
    model_name: t.Optional[str] = None,
    model_version: t.Optional[int] = None,
    model_alias: t.Optional[str] = None,
    run_id: t.Optional[str] = None,
    model_path: t.Optional[str] = None,
) -> str:
    """
    Get an mlflow model's URI based on its name, version, alias, path, and/or run id.

    References:
        - https://docs.databricks.com/gcp/en/mlflow/models
        - https://www.mlflow.org/docs/latest/concepts.html#artifact-locations
    """
    if run_id is not None and model_path is not None:
        return f"runs:/{run_id}/{model_path}"
    elif model_name is not None and model_version is not None:
        return f"models:/{model_name}/{model_version}"
    elif model_name is not None and model_alias is not None:
        return f"models:/{model_name}@{model_alias}"
    else:
        raise ValueError(
            "unable to determine model URI from inputs: "
            f"{model_name=}, {model_version=}, {model_alias=}, {model_path=}, {run_id=}"
        )
