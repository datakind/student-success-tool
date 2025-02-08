"""Cloud storage sftp server get related helper functions.
"""

import paramiko
from google.cloud import storage
from pydantic import BaseModel


def get_sftp_bucket_name(env_var: str) -> str:
    return env_var.lower() + "_sftp_ingestion"


# For functionality that interfaces with GCS, wrap it in a class for easier mock unit testing.
class StorageControl(BaseModel):
    def copy_from_sftp_to_gcs(
        self,
        sftp_host: str,
        sftp_port: int,
        sftp_user: str,
        sftp_password: str,
        sftp_file: str,
        bucket_name: str,
        blob_name: str,
    ) -> None:
        """Copies a file from an SFTP server to a GCS bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        with paramiko.Transport((sftp_host, sftp_port)) as transport:
            transport.connect(username=sftp_user, password=sftp_password)
            client = paramiko.SFTPClient.from_transport(transport)
            with client.open(sftp_file) as f:
                blob.upload_from_file(f)

    def create_bucket_if_not_exists(
        self,
        bucket_name: str,
    ) -> None:
        """Copies a file from an SFTP server to a GCS bucket."""
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
        bucket.storage_class = "STANDARD"
        new_bucket = storage_client.create_bucket(bucket, location="us")
