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
    checkpoint_config: dict,
    target_config: dict,
) -> str:
    """
    Generate a standard model name from configuration target & checkpoint components.
    Required components will be included and if missing, will raise a 'KeyError', while
    optional components are omitted.

    Format:
    "{institution_id}_{target.category}_[T_{target.unit}{target.value}]_C_{checkpoint.unit}{checkpoint.value}_[{checkpoint.optional_desc}]"
    """
    # Build model name components, skipping any optional values
    parts = [
        institution_id,
        target_config["category"],
        f"T_{target_config['value']}{target_config['unit']}"
            if target_config.get("value") and target_config.get("unit") else None,
        f"C_{checkpoint_config.get('value')}{checkpoint_config.get('unit')}",
        checkpoint_config.get("optional_desc")
    ]

    # Filter out None or empty strings to avoid extra underscores
    return "_".join(filter(None, parts))


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
