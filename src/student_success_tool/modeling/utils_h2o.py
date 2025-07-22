import os
import tempfile
import mlflow
import h2o
from h2o.model.model_base import ModelBase
from mlflow.artifacts import download_artifacts

def download_model_artifact(run_id: str, artifact_subdir: str = "model") -> str:
    """
    Downloads a model directory artifact from MLflow and returns the local path.

    Args:
        run_id: MLflow run ID.
        artifact_subdir: Subdirectory in the run artifacts, usually 'model'.

    Returns:
        Path to the downloaded model directory.
    """
    local_dir = tempfile.mkdtemp()
    artifact_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_subdir, dst_path=local_dir
    )
    return artifact_path  # already includes artifact_subdir


def load_h2o_model(run_id: str, artifact_path: str = "model") -> ModelBase:
    """
    Initializes H2O, downloads the model from MLflow, and loads it.
    Basically a wrapper for H2O's `h2o.load_model` function.

    Args:
        run_id: MLflow run ID containing the H2O model.
        artifact_path: Path inside run artifacts where model was logged.

    Returns:
        Loaded H2O model object.
    """
    if not h2o.connection():
        h2o.init()

    model_dir = download_model_artifact(run_id=run_id, artifact_subdir=artifact_path)
    return h2o.load_model(model_dir)
