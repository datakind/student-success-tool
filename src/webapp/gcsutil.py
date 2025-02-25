"""Cloud storage related helper functions.
"""

import datetime
from pydantic import BaseModel

from google.cloud import storage, storage_control_v2
from google.cloud.storage import Client
from google.cloud.storage_control_v2 import StorageControlClient
from typing import Any
from .config import gcs_vars, databricks_vars
from .validation import validate_file_reader, SchemaType
from .utilities import (
    SchemaType,
)
import google.auth
from google.auth.transport import requests


credentials, project_id = google.auth.default()
SIGNED_URL_EXPIRY_MIN = 30


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


# Wrapping the usages in a class makes it easier to unit test via mocks.


# Wrapping the usages in a class makes it easier to unit test via mocks.
class StorageControl(BaseModel):
    """Object to manage interfacing with GCS."""

    def generate_upload_signed_url(self, bucket_name: str, file_name: str) -> str:
        """Generates a v4 signed URL for uploading a blob using HTTP PUT."""
        r = requests.Request()
        credentials.refresh(r)
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

        service_account_email = ""
        if hasattr(credentials, "service_account_email"):
            service_account_email = credentials.service_account_email
        url = blob.generate_signed_url(
            version="v4",
            service_account_email=service_account_email,
            access_token=credentials.token,
            # How long the url is usable for.
            expiration=datetime.timedelta(minutes=SIGNED_URL_EXPIRY_MIN),
            # Allow PUT requests using this URL.
            method="PUT",
            content_type="text/csv",
        )

        return url

    def generate_download_signed_url(self, bucket_name: str, blob_name: str) -> str:
        """Generates a v4 signed URL for downloading a blob using HTTP GET."""
        r = requests.Request()
        credentials.refresh(r)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            raise ValueError("Storage bucket not found.")
        blob = bucket.blob(blob_name)
        if not blob.exists():
            raise ValueError(blob_name + ": File not found.")
        service_account_email = ""
        if hasattr(credentials, "service_account_email"):
            service_account_email = credentials.service_account_email
        url = blob.generate_signed_url(
            version="v4",
            service_account_email=service_account_email,
            access_token=credentials.token,
            # How long the url is usable for.
            expiration=datetime.timedelta(minutes=SIGNED_URL_EXPIRY_MIN),
            # Allow GET requests using this URL.
            method="GET",
        )
        return url

    def delete_bucket(self, bucket_name: str) -> None:
        storage_client = storage.Client()
        # Delete the GCS bucket.  Force=True handles non-empty buckets.
        bucket = storage_client.get_bucket(bucket_name)
        bucket.delete(force=True)

    def create_bucket(self, bucket_name: str) -> None:
        """
        Create a new bucket in the US region with the standard storage
        class.
        """
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        if bucket.exists():
            raise ValueError(bucket_name + " already exists. Creation failed.")
        # Update with URL?
        # fmt: off
        bucket.cors = [
            {
                "origin": ["*"],
                "responseHeader": ["*"],
                "method": ["GET", "OPTIONS", "PUT", "POST"],
                "maxAgeSeconds": 3600
            }
        ]
        # fmt: on
        # Apply TTL to unvalidated files. This may occur if an API caller uploads but doesn't call validate.
        # fmt: off
        bucket.lifecycle_rules = [
            {
            "action": {"type": "Delete"},
            "condition": {"age": 1, "matchesPrefix": ["unvalidated/"]}
            }
        ]
        # fmt: on
        bucket.storage_class = "STANDARD"
        # Grant object admin access to the specified service account.
        new_bucket = storage_client.create_bucket(
            bucket, location=gcs_vars["GCP_REGION"]
        )
        policy = new_bucket.get_iam_policy(requested_policy_version=3)
        policy.bindings.append(
            {
                "role": "roles/storage.objectAdmin",
                # The account triggering the job is not the same as the account reading the buckets content INSIDE the job. This is the account reading the buckets from within Databricks accounts.
                "members": {
                    "serviceAccount:"
                    + databricks_vars["DATABRICKS_SERVICE_ACCOUNT_EMAIL"]
                },
            }
        )
        new_bucket.set_iam_policy(policy)

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
        return res

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
        if not blob.exists():
            raise ValueError(file_name + ": File not found.")
        blob.download_to_filename(destination_file_name)

    def move_file(self, bucket_name: str, prev_name: str, new_name: str):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            raise ValueError("Storage bucket not found.")
        blob = bucket.blob(prev_name)
        if not blob.exists():
            raise ValueError(prev_name + ": File not found.")
        new_blob = bucket.blob(new_name)
        if new_blob.exists():
            raise ValueError(new_name + ": File already exists.")
        bucket.copy_blob(blob, bucket, new_name)
        blob.delete()

    def delete_file(self, bucket_name: str, file_name: str):
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            raise ValueError("Storage bucket not found.")
        blob = bucket.blob(file_name)
        if not blob.exists():
            raise ValueError(file_name + ": File not found.")
        blob.delete()

    def validate_file(
        self, bucket_name: str, file_name: str, allowed_schemas: set[SchemaType]
    ) -> set[SchemaType]:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"unvalidated/{file_name}")
        new_blob_name = f"validated/{file_name}"
        schems = set()
        try:
            with blob.open("r") as file:
                schems = validate_file_reader(file, allowed_schemas)
        except Exception as e:
            blob.delete()
            raise e
        new_blob = bucket.blob(new_blob_name)
        if new_blob.exists():
            raise ValueError(new_blob_name + ": File already exists.")
        bucket.copy_blob(blob, bucket, new_blob_name)
        blob.delete()
        return schems

    def get_file_contents(self, bucket_name: str, file_name: str):
        """Returns file as a bytes object."""
        print("aaaaaaaaaaaaaaaaaaaaaaaaaa1")
        storage_client = storage.Client()
        print("aaaaaaaaaaaaaaaaaaaaaaaaaa2")
        bucket = storage_client.get_bucket(bucket_name)
        print("aaaaaaaaaaaaaaaaaaaaaaaaaa3")
        blob = bucket.blob(file_name)
        print("aaaaaaaaaaaaaaaaaaaaaaaaaa4")
        res = blob.download_as_bytes()
        print("aaaaaaaaaaaaaaaaaaaaaaaaaa5")
        return res
