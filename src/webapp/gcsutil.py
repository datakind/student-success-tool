"""Cloud storage related helper functions.
"""

import datetime

from google.cloud import storage, storage_control_v2
from google.cloud.storage import Client
from typing import Any

def generate_upload_signed_url(bucket_name: str, blob_name: str) -> str:
    """Generates a v4 signed URL for uploading a blob using HTTP PUT."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        raise ValueError("Storage bucket not found.")
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise ValueError("Blob not found.")

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow PUT requests using this URL.
        method="PUT",
        content_type="text/csv",
    )

    return url


def generate_download_signed_url(bucket_name: str, blob_name: str) -> str:
    """Generates a v4 signed URL for uploading a blob using HTTP PUT."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if not bucket.exists():
        raise ValueError("Storage bucket not found.")
    blob = bucket.blob(blob_name)
    if not blob.exists():
        raise ValueError("Blob not found.")

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow GET requests using this URL.
        method="GET",
        content_type="text/csv",
    )

    return url


def create_bucket(bucket_name: str) -> Any:
    """
    Create a new bucket in the US region with the standard storage
    class.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if bucket.exists():
        raise ValueError(bucket_name + " already exists. Creation failed.")
    bucket.storage_class = "STANDARD"
    new_bucket = client.create_bucket(bucket, location="us")
    return new_bucket


def create_folders(bucket_name: str, folder_names: list[str]) -> None:
    """
    Create a list of new folders in a GCS bucket.
    """

    storage_control_client = storage_control_v2.StorageControlClient()
    # The storage bucket path uses the global access pattern, in which the "_"
    # denotes this bucket exists in the global namespace.
    project_path = storage_control_client.common_project_path("_")
    bucket_path = f"{project_path}/buckets/{bucket_name}"

    for f in folder_names:
        request = storage_control_v2.CreateFolderRequest(
            parent=bucket_path,
            folder_id=f,
        )
        response = storage_control_client.create_folder(request=request)


def list_folders(bucket_name: str, folder_names: list[str]) -> None:
    """
    Create a list of new folders in a GCS bucket.
    """

    storage_control_client = storage_control_v2.StorageControlClient()
    # The storage bucket path uses the global access pattern, in which the "_"
    # denotes this bucket exists in the global namespace.
    project_path = storage_control_client.common_project_path("_")
    bucket_path = f"{project_path}/buckets/{bucket_name}"

    for f in folder_names:
        request = storage_control_v2.CreateFolderRequest(
            parent=bucket_path,
            folder_id=f,
        )
        response = storage_control_client.create_folder(request=request)


def list_blobs_in_folder(bucket_name: str, prefix: str, delimiter=None) -> list[str]:
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        a/1.txt
        a/b/2.txt

    If you specify prefix ='a/', without a delimiter, you'll get back:

        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a/' and delimiter='/', you'll get back
    only the file directly under 'a/':

        a/1.txt

    As part of the response, you'll also get back a blobs.prefixes entity
    that lists the "subfolders" under `a/`:

        a/b/
    """

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    # Note: The call returns a response only when the iterator is consumed.
    res = []
    for blob in blobs:
        res.append(blob.name)

    if delimiter:
        for prefix in blobs.prefixes:
            res.append(prefix)


def download_file(bucket_name: str, file_name: str, destination_file_name: str):
    """Downloads a blob from the bucket."""

    # The path to which the file should be downloaded
    # destination_file_name = "local/path/to/file"

    client = storage.Client()

    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        raise ValueError("Storage bucket not found.")

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(file_name)
    blob.download_to_filename(destination_file_name)
