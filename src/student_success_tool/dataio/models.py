import logging
import typing as t

import mlflow

LOGGER = logging.getLogger(__name__)


def load_mlflow_model(
    model_uri: str,
    framework: t.Optional[t.Literal["sklearn", "xgboost", "lightgbm"]] = None,
) -> object:
    """
    Load a (registered) MLFlow model of whichever model type from a specified URI.

    Args:
        model_uri
        framework

    References:
        - https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html#mlflow.sklearn.load_model
        - https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model
    """
    load_model_func = (
        mlflow.sklearn.load_model
        if framework == "sklearn"
        else mlflow.xgboost.load_model
        if framework == "xgboost"
        else mlflow.lightgbm.load_model
        if framework == "lightgbm"
        else mlflow.pyfunc.load_model
    )
    model = load_model_func(model_uri)
    LOGGER.info("mlflow model loaded from '%s'", model_uri)
    return model
