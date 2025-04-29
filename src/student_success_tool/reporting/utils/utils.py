import os 
import sys
import shutil
import typing as t
import mlflow

import contextlib
import pathlib

def embed_image(description: str, local_path: str | pathlib.Path, width: int) -> str:
    return f'<img src="{os.path.relpath(local_path, start=os.getcwd())}" alt="{description}" width="{width}">'

def list_paths_in_directory(run_id: str, directory: str) -> t.List[str]:
    """
    List all artifact paths inside a specific directory for a run_id.
    Only retrieves immediate contents (non-recursive).
    
    Args:
        run_id (str): The MLflow run ID.
        directory (str): The subfolder path (relative to run root).
        
    Returns:
        List[str]: A list of file or subfolder paths (relative to run root).
    """
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path=directory)
    return [artifact.path for artifact in artifacts]

def download_artifact(run_id: str, description: str, artifact_path: str, width: int, local_folder: str) -> str:
    os.makedirs(local_folder, exist_ok=True)
    local_path = safe_mlflow_download_artifacts(run_id, artifact_path, local_folder)
    return embed_image(description, local_path, width)

def download_static_asset(description: str, static_path: pathlib.Path, width: int, local_folder: str) -> str:
    os.makedirs(local_folder, exist_ok=True)
    dst_path = os.path.join(local_folder, static_path.name)
    shutil.copy(static_path, dst_path)
    return embed_image(description, dst_path, width)

def safe_mlflow_download_artifacts(run_id: str, artifact_path: str, dst_path: str) -> str:
    @contextlib.contextmanager
    def suppress_display():
        try:
            import IPython.display
            old_display = IPython.display.display
            IPython.display.display = lambda *args, **kwargs: None
            yield
        finally:
            if 'IPython' in sys.modules:
                IPython.display.display = old_display

    with suppress_display():
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=dst_path,
        )
    return local_path