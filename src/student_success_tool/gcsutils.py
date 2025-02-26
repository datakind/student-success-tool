"""
Utility functions that interact with Google Cloud Storage.
For instance, copying files from Databricks volumes into GCP buckets.
"""

from google.cloud import storage
from databricks.sdk.runtime import dbutils  # noqa: F401


def save_file(bucket: object, src_volume_filepath: str, dest_bucket_pathname: str):
    """Save file from databricks volume to a GCP bucket path."""
    blob = bucket.blob(dest_bucket_pathname)
    if blob.exists():
        raise ValueError(dest_bucket_pathname + ": File already exists in bucket.")
    blob.upload_from_filename(src_volume_filepath)


def get_filename_from_file_path(file_path: str):
    """Get the filename only of a given filepath, e.g. for home/directory/file.csv, return just file.csv."""
    return file_path.split("/")[-1]


def publish_output_files(
    db_workspace: str,
    institution_name: str,
    external_bucket_name: str,
    sst_job_id: str,
    approved: bool,
) -> str:
    """
    Publish output files to bucket, with folder determined based on approved parameter,
    and return the status of the job as a string.
    """
    # Destination: The output directory in the bucket that the files get moved to.
    bucket_directory = f"{'approved' if approved else 'unapproved'}/{sst_job_id}"
    # Source: The Databricks volume path we want to read the files from.
    volume_path_top_level = f"/Volumes/{db_workspace}/{institution_name}_gold/gold_volume/inference_jobs/{sst_job_id}/ext/"
    volume_path_inference_folder = f"{volume_path_top_level}/inference_output"

    storage_client = storage.Client()
    bucket = storage_client.bucket(external_bucket_name)

    # Full filepaths for the Databricks volume.
    files_to_move = []
    status_string = ""
    # There will be other files in the inference_output folder generated by Databricks
    # including time started/completed, and completion status.
    for f in dbutils.fs.ls(volume_path_inference_folder):
        filename = f.name
        if filename.startswith("_"):
            if not filename.startswith("_committed") and not filename.startswith(
                "_started"
            ):
                status_string = filename
        elif filename.endswith(".csv"):
            files_to_move.append(f"{volume_path_inference_folder}/{filename}")
    # The top level volume directory can include other files like pngs etc. that are part
    # of the published set of files. So we still need to iterate over these files.
    for f in dbutils.fs.ls(volume_path_top_level):
        files_to_move.append(f"{volume_path_top_level}/{f.name}")

    for f in files_to_move:
        save_file(
            bucket,
            f,
            f"{bucket_directory}/{get_filename_from_file_path(f)}",
        )
    return status_string
