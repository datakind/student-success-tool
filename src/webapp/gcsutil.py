"""Cloud storage related helper functions.
"""

import datetime
from pydantic import BaseModel

from google.cloud import storage, storage_control_v2
from google.cloud.storage import Client
from google.cloud.storage_control_v2 import StorageControlClient
from typing import Any


def rename_file(
    bucket_name,
    file_name,
    new_file_name,
):
    """Moves a blob from one bucket to another with a new name."""
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(file_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    # There is also an `if_source_generation_match` parameter, which is not used in this example.
    destination_generation_match_precondition = 0

    blob_copy = source_bucket.copy_blob(
        source_blob,
        new_file_name,
        if_generation_match=destination_generation_match_precondition,
    )
    source_bucket.delete_blob(file_name)


def generate_upload_signed_url(client: Client, bucket_name: str, file_name: str) -> str:
    """Generates a v4 signed URL for uploading a blob using HTTP PUT."""
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        raise ValueError("Storage bucket not found.")
    for prefix in ("unvalidated/", "validated/"):
        blob_name = prefix + file_name
        blob = bucket.blob(blob_name)
        if blob.exists():
            raise ValueError("File already exists.")
    # All files uploaded directly are considered unvalidated.
    blob_name = "unvalidated/" + file_name
    blob = bucket.blob(blob_name)
    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow PUT requests using this URL.
        method="PUT",
        content_type="text/csv",
    )

    return url


def create_bucket(client: Client, bucket_name: str) -> None:
    """
    Create a new bucket in the US region with the standard storage
    class.
    """
    bucket = client.bucket(bucket_name)
    if bucket.exists():
        raise ValueError(bucket_name + " already exists. Creation failed.")
    bucket.storage_class = "STANDARD"
    new_bucket = client.create_bucket(bucket, location="us")


# Wrapping the usages in a class makes it easier to unit test via mocks.


# Wrapping the usages in a class makes it easier to unit test via mocks.
class StorageControl(BaseModel):
    """Object to manage interfacing with GCS."""

    def generate_upload_signed_url(self, bucket_name: str, file_name: str) -> str:
        """Generates a v4 signed URL for uploading a blob using HTTP PUT."""
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            raise ValueError("Storage bucket not found.")
        for prefix in ("unvalidated/", "validated/"):
            blob_name = prefix + file_name
            blob = bucket.blob(blob_name)
            if blob.exists():
                raise ValueError("File already exists.")
        # All files uploaded directly are considered unvalidated.
        blob_name = "unvalidated/" + file_name
        blob = bucket.blob(blob_name)
        url = blob.generate_signed_url(
            version="v4",
            # This URL is valid for 15 minutes
            expiration=datetime.timedelta(minutes=15),
            # Allow PUT requests using this URL.
            method="PUT",
            content_type="text/csv",
        )

        return url

    def generate_download_signed_url(self, bucket_name: str, blob_name: str) -> str:
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

    def create_bucket(self, bucket_name: str) -> None:
        """
        Create a new bucket in the US region with the standard storage
        class.
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        if bucket.exists():
            raise ValueError(bucket_name + " already exists. Creation failed.")
        bucket.storage_class = "STANDARD"
        new_bucket = storage_client.create_bucket(bucket, location="us")

    def list_blobs_in_folder(
        self, bucket_name: str, prefix: str, delimiter=None
    ) -> list[str]:
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
        blobs = storage_client.list_blobs(
            bucket_name, prefix=prefix, delimiter=delimiter
        )

        # Note: The call returns a response only when the iterator is consumed.
        res = []
        for blob in blobs:
            res.append(blob.name)

        if delimiter:
            for prefix in blobs.prefixes:
                res.append(prefix)

    def download_file(
        self, bucket_name: str, file_name: str, destination_file_name: str
    ):
        """Downloads a blob from the bucket."""

        # The path to which the file should be downloaded
        # destination_file_name = "local/path/to/file"
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            raise ValueError("Storage bucket not found.")

        # Construct a client side representation of a blob.
        # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
        # any content from Google Cloud Storage. As we don't need additional data,
        # using `Bucket.blob` is preferred here.
        blob = bucket.blob(file_name)
        blob.download_to_filename(destination_file_name)
