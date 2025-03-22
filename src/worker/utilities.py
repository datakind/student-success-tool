"""Cloud storage sftp server get related helper functions.
"""

import paramiko
from google.cloud import storage
from pydantic import BaseModel
import os
import stat
import datetime
import csv
import io
from collections import defaultdict

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

    def list_sftp_files(self, sftp_host: str, sftp_port: int, sftp_user: str, sftp_password: str, remote_path: str = ".") -> list:
        """
        Connects to an SFTP server and recursively lists all files under the given remote_path.
        For each file, it returns the file path, size (in bytes), and last modified date (ISO format).
        
        Args:
            sftp_host (str): SFTP server hostname.
            sftp_port (int): SFTP server port.
            sftp_user (str): SFTP username.
            sftp_password (str): SFTP password.
            remote_path (str): Remote directory to start listing from (default is ".").
            
        Returns:
            list: A list of dictionaries for each file with keys 'path', 'size', and 'modified'.
        """
        file_list = []

        with paramiko.Transport((sftp_host, sftp_port)) as transport:
            transport.connect(username=sftp_user, password=sftp_password)
            with paramiko.SFTPClient.from_transport(transport) as sftp:

                def recursive_list(path: str):
                    for attr in sftp.listdir_attr(path):
                        entry_path = os.path.join(path, attr.filename)
                        if stat.S_ISDIR(attr.st_mode):
                            recursive_list(entry_path)
                        else:
                            # Convert the modification time (epoch) to an ISO-formatted date.
                            modified_iso = datetime.datetime.fromtimestamp(attr.st_mtime).isoformat()
                            size_mb = round(attr.st_size / (1024 * 1024), 2)
                            file_info = {
                                "path": entry_path,
                                "size": size_mb,
                                "modified": modified_iso,
                            }
                            file_list.append(file_info)
                recursive_list(remote_path)

        return file_list

    def create_bucket_if_not_exists(
        self,
        bucket_name: str,
    ) -> None:
        """Copies a file from an SFTP server to a GCS bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        if bucket.exists():
            print(f"Bucket '{bucket_name}' already exists.")
            return bucket
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

        return new_bucket

def sftp_split_files(storage_control: StorageControl, bucket_name: str, source_blob_name: str, destination_folder: str, institution_column: str = "institution_id"):
    """
    Splits a CSV blob by the values in the given institution column and writes each subset
    as a new CSV blob in the specified destination folder within the same bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The path/name of the source CSV blob.
        destination_folder (str): The folder (prefix) within the bucket where split CSVs will be stored.
        institution_column (str): The CSV column name to use for splitting. Default is "institution_id".

    Returns:
        list[str]: A list of new blob names created in the bucket.
    
    Raises:
        ValueError: If the source blob is not found.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Retrieve the source blob
    source_blob = bucket.blob(source_blob_name)
    if not source_blob.exists():
        raise ValueError(f"Source blob '{source_blob_name}' not found in bucket '{bucket_name}'.")

    # Download CSV content as text
    csv_content = source_blob.download_as_text()
    csv_file = io.StringIO(csv_content)
    
    reader = csv.DictReader(csv_file)
    header = reader.fieldnames
    if institution_column not in header:
        raise ValueError(f"Institution column '{institution_column}' not found in CSV header.")
    
    # Group rows by institution id
    groups = defaultdict(list)
    for row in reader:
        groups[row[institution_column]].append(row)
    
    new_blob_names = []
    # For each institution id, create a new CSV content and upload as a new blob.
    for inst_id, rows in groups.items():
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
        # Construct a blob name using the destination folder and the institution id.
        new_blob_name = f"{destination_folder}/{institution_column}_{inst_id}.csv"
        new_blob = bucket.blob(new_blob_name)
        new_blob.upload_from_string(output.getvalue(), content_type="text/csv")
        new_blob_names.append(new_blob_name)
    
    return new_blob_names