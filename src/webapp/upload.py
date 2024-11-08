import datetime

from google.cloud import storage


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
