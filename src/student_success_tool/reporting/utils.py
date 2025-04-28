import os 
import mlflow
import shutil
import contextlib
import sys
import pathlib

def embed_image(description: str, local_path: str | pathlib.Path, width: int) -> str:
    return f'<img src="{os.path.relpath(local_path, start=os.getcwd())}" alt="{description}" width="{width}">'

def download_artifact(run_id: str, description: str, artifact_path: str, width: int, local_folder: str) -> str:
    os.makedirs(local_folder, exist_ok=True)
    local_path = safe_mlflow_download_artifacts(run_id, artifact_path, local_folder)
    return embed_image(description, local_path, width)

def download_static_asset(description: str, static_path: pathlib.Path, width: int, local_folder: str) -> str:
    os.makedirs(local_folder, exist_ok=True)
    dst_path = os.path.join(local_folder, static_path.name)
    shutil.copy(static_path, dst_path)

    return embed_image(description, dst_path, width)

def safe_mlflow_download_artifacts(run_id: str, artifact_path: str, dst_path: str):
    @contextlib.contextmanager
    def suppress_display_errors():
        with open(os.devnull, "w") as fnull:
            old_stdout = sys.stdout
            sys.stdout = fnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    with suppress_display_errors():
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=artifact_path,
            dst_path=dst_path,
        )
    return local_path