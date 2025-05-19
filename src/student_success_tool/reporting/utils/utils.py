import os
import shutil
import typing as t
import logging
import mlflow
import pathlib
from importlib.abc import Traversable
from importlib.resources import as_file

LOGGER = logging.getLogger(__name__)


def download_artifact(
    run_id: str,
    local_folder: str,
    artifact_path: str,
    description: t.Optional[str] = None,
    fixed_width: str = "125mm",
) -> str:
    """
    Downloads artifact from MLflow run using mlflow.artifacts.download_artifacts(...) and
    returns the path. This method can be used for images, csv, and other files.

    Args:
        run_id: MLflow run ID
        local_folder: Local folder to download artifact to
        artifact_path: Path to artifact
        description: Description of the image
        fixed_width: Desired fixed width (e.g., "150mm", "100px").

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
        return embed_image(description, local_path)
    else:
        return local_path


def download_static_asset(
    description: str,
    static_path: Traversable,
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
        return embed_image(description, dst_path, fixed_width="40mm", alignment="left")
    else:
        return dst_path


def log_card(local_path: str, run_id: str) -> None:
    """
    Logs card as an ML artifact in the run.

    Args:
        local_path: Path to model card PDF
    """
    with mlflow.start_run(run_id=run_id) as run:
        mlflow.log_artifact(local_path, "model_card")
        LOGGER.info(f"Logged model card PDF as an ML artifact at '{run_id}'")


def embed_image(
    description: str,
    local_path: t.Optional[str | pathlib.Path],
    fixed_width: str = "125mm",
    alignment: str = "center",
) -> str:
    """
    Embeds image in markdown with inline CSS to control rendering in WeasyPrint.

    Args:
        description: Description of the image.
        local_path: Path to the image file.
        fixed_width: Desired fixed width (e.g., "150mm", "100px").
        alignment: Horizontal alignment ("left", "right", "center").

    Returns:
        Inline HTML string to be embedded in markdown.
    """
    local_path_str = str(local_path)
    rel_path = os.path.relpath(local_path_str, start=os.getcwd())

    alignment = alignment.lower()
    if alignment == "left":
        css_alignment = "display: block; margin-left: 0; margin-right: auto;"
    elif alignment == "right":
        css_alignment = "display: block; margin-left: auto; margin-right: 0;"
    else:
        css_alignment = "display: block; margin: auto;"

    style = f"{css_alignment} width: {fixed_width}; height: auto; max-width: 100%;"

    return (
        f'<img src="{rel_path}" alt="{description}" style="{style}">'
    )


def list_paths_in_directory(run_id: str, directory: str) -> t.List[str]:
    """
    List all artifact paths inside a specific directory for a run_id.
    Only retrieves immediate contents (non-recursive).

    Args:
        run_id: The MLflow run ID.
        directory: The subfolder path (relative to run root).

    Returns:
        A list of file or subfolder paths (relative to run root).
    """
    artifacts = mlflow.artifacts.list_artifacts(run_id=run_id, artifact_path=directory)
    return [artifact.path for artifact in artifacts]
