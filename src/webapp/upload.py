"""Cloud storage upload related helper functions.
"""

import datetime

from google.cloud import storage
from typing import Any


def generate_upload_signed_url(bucket_name: str, blob_name: str) -> str:
    """Generates a v4 signed URL for uploading a blob using HTTP PUT."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
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


def create_bucket(bucket_name: str) -> Any:
    """
    Create a new bucket in the US region with the standard storage
    class.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.storage_class = "STANDARD"
    new_bucket = storage_client.create_bucket(bucket, location="us")
    return new_bucket
