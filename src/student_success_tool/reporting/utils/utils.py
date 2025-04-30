import os 
import shutil
import typing as t
import mlflow
import pathlib
from importlib.abc import Traversable
from importlib.resources import as_file


def download_artifact(
    run_id: str,
    local_folder: str,
    artifact_path: str,
    width: t.Optional[int] = None,
    description: t.Optional[str] = None,
) -> str:
    """
    Downloads artifact from MLflow run using mlflow.artifacts.download_artifacts(...) and
    returns the path. This method can be used for images, csv, and other files.

    Args:
        run_id: MLflow run ID
        local_folder: Local folder to download artifact to
        artifact_path: Path to artifact
        width: Width of image in pixels
        description: Description of the image

    Returns:
        Local path to artifact OR inline HTML string with path information if image
    """
    os.makedirs(local_folder, exist_ok=True)

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=local_folder,
    )

    if local_path.lower().endswith((".png", ".jpg", ".jpeg")):
        if description is None:
            description = os.path.basename(local_path)
        return embed_image(description, local_path, width)
    else:
        return local_path


def download_static_asset(
    description: str,
    static_path: Traversable,
    width: int,
    local_folder: str,
) -> str:
    """
    Downloads static asset from local folder and returns the path. This method
    does not use mlflow and is not associated with an mlflow run. This method
    can be used for images, csv, and other files. We primarily utilize this
    method to download the DataKind logo and this can be used for any static
    assets that will go in all model cards.

    Args:
        description: Description of the image
        static_path: Path to static asset
        width: Width of image in pixels
        local_folder: Local folder to download artifact to

    Returns:
        Local path to artifact OR inline HTML string with path information if image
    """
    os.makedirs(local_folder, exist_ok=True)

    dst_path = os.path.join(local_folder, static_path.name)

    with as_file(static_path) as actual_path:
        shutil.copy(actual_path, dst_path)

    if dst_path.lower().endswith((".png", ".jpg", ".jpeg")):
        if description is None:
            description = os.path.basename(dst_path)
        return embed_image(description, dst_path, width)
    else:
        return dst_path


def embed_image(
    description: str,
    local_path: t.Optional[str | pathlib.Path],
    width: t.Optional[int | None] = 400,
) -> str:
    """
    Embeds image in markdown by returning inline HTML to accomodate for flexibility with
    image size and name.

    Args:
        description: Description of the image
        local_path: Path to image
        width: Width of image in pixels

    Returns:
        str: inline HTML string to be embedded in markdown
    """
    local_path_str = str(local_path)
    rel_path = os.path.relpath(local_path_str, start=os.getcwd())
    return f'<img src="{rel_path}" alt="{description}" width="{width}">'


def list_paths_in_directory(run_id: str, directory: str) -> t.List[str]:
    """
    List all artifact paths inside a specific directory for a run_id.
    Only retrieves immediate contents (non-recursive).

    Args:
        run_id: The MLflow run ID.
        directory: The subfolder path (relative to run root).

    Returns:
        List[str]: A list of file or subfolder paths (relative to run root).
    """
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path=directory)
    return [artifact.path for artifact in artifacts]
