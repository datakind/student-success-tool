import importlib.metadata
import logging
import os
import shutil
import tempfile
import typing as t
from collections.abc import Collection

import mlflow
import mlflow.artifacts
import mlflow.tracking
import yaml  # type: ignore[import-untyped]

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


def patch_mlflow_model_env_files(
    run_id: str,
    packages: Collection[str] = ("mlflow", "pandas", "scikit-learn"),
    *,
    force: bool = True,
    mlflow_client: mlflow.tracking.MlflowClient,
) -> None:
    """
    Patch an mlflow model's environment metadata with the versions of packages
    present in the current Python environment.

    Args:
        run_id
        packages
        force
        mlflow_client

    References:
        - https://docs.databricks.com/aws/en/machine-learning/automl/regression-train-api#no-module-named-pandascoreindexesnumeric
    """
    package_versions = {
        package: importlib.metadata.version(package) for package in packages
    }
    # set up local dir for downloading the artifacts
    tmp_dir = str(tempfile.TemporaryDirectory().name)
    os.makedirs(tmp_dir)
    # fix conda.yaml file's pip dependencies
    conda_fpath = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model/conda.yaml", dst_path=tmp_dir
    )
    with open(conda_fpath) as f:
        conda_env = yaml.load(f, Loader=yaml.FullLoader)
    conda_env["dependencies"][-1]["pip"] = _patch_libs_in_env_file(
        package_versions,
        conda_env["dependencies"][-1]["pip"],
        force=force,
        fname="conda.yaml",
    )
    with open(f"{tmp_dir}/conda.yaml", "w") as f:
        f.write(yaml.dump(conda_env))
    mlflow_client.log_artifact(
        run_id=run_id, local_path=conda_fpath, artifact_path="model"
    )
    # fix requirements.txt file's pip dependencies
    venv_fpath = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{run_id}/model/requirements.txt", dst_path=tmp_dir
    )
    with open(venv_fpath) as f:
        venv_libs = [lib.strip() for lib in f.readlines()]
    venv_libs = _patch_libs_in_env_file(
        package_versions, venv_libs, force=force, fname="requirements.txt"
    )
    with open(f"{tmp_dir}/requirements.txt", "w") as f:
        f.write("\n".join(venv_libs))
    mlflow_client.log_artifact(
        run_id=run_id, local_path=venv_fpath, artifact_path="model"
    )
    shutil.rmtree(tmp_dir)


def _patch_libs_in_env_file(
    package_versions: dict[str, str], libs: list[str], *, force: bool, fname: str
) -> list[str]:
    for package, version in package_versions.items():
        package_exists = any(lib.startswith(f"{package}==") for lib in libs)
        if not package_exists:
            libs.append(f"{package}=={version}")
            LOGGER.info("adding %s==%s to %s", package, version, fname)
        elif force is True:
            libs = [lib for lib in libs if not lib.startswith(f"{package}==")]
            libs.append(f"{package}=={version}")
            LOGGER.info("overwriting %s==%s in %s", package, version, fname)
    return libs
