import os
import tempfile
import mlflow
import h2o
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


def load_h2o_model(run_id: str, artifact_path: str = "model"):
    """
    Initializes H2O, downloads the model artifact from MLflow, and loads it.
    Cleans up the temp directory after loading.
    """
    if not h2o.connection():
        h2o.init()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_model_dir = download_artifacts(
            run_id=run_id, artifact_path=artifact_path, dst_path=tmp_dir
        )

        # Find the actual model file inside the directory
        files = os.listdir(local_model_dir)
        if not files:
            raise FileNotFoundError(f"No model file found in {local_model_dir}")

        model_path = os.path.join(local_model_dir, files[0])
        return h2o.load_model(model_path)
