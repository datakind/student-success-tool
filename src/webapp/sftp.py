"""Cloud storage sftp server get related helper functions.
"""

import paramiko
from google.cloud import storage


def copy_from_sftp_to_gcs(
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
