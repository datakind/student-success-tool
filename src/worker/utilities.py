"""Cloud storage sftp server get related helper functions."""

import paramiko
from google.cloud import storage
from pydantic import BaseModel
import os
import stat
from datetime import datetime, timedelta
import csv
import io
from collections import defaultdict
import logging
from typing import List, Dict, Any
import requests
import pandas as pd
import io
import re

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
            if client is None:
                raise RuntimeError("Failed to create SFTP client.")
            # Open the file in binary read mode.
            with client.open(sftp_file, "rb") as f:
                blob.upload_from_file(f)

    def list_sftp_files(
        self,
        sftp_host: str,
        sftp_port: int,
        sftp_user: str,
        sftp_password: str,
        remote_path: str = ".",
    ) -> List[Dict[str, Any]]:
        """
        Connects to an SFTP server and recursively lists all files under the given remote_path.
        For each file, it returns the file path, size (in MB), and last modified date (ISO format).

        Args:
            sftp_host (str): SFTP server hostname.
            sftp_port (int): SFTP server port.
            sftp_user (str): SFTP username.
            sftp_password (str): SFTP password.
            remote_path (str): Remote directory to start listing from (default is ".").

        Returns:
            List[Dict[str, Any]]: A list of dictionaries for each file with keys 'path', 'size', and 'modified'.
        """
        file_list: List[Dict[str, Any]] = []

        with paramiko.Transport((sftp_host, sftp_port)) as transport:
            transport.connect(username=sftp_user, password=sftp_password)
            sftp = paramiko.SFTPClient.from_transport(transport)
            if sftp is None:
                raise RuntimeError("Failed to create SFTP client.")

            def recursive_list(path: str) -> None:
                for attr in sftp.listdir_attr(path):
                    entry_path = os.path.join(path, attr.filename)
                    # Ensure attr.st_mode is an int.
                    if attr.st_mode is not None and stat.S_ISDIR(attr.st_mode):
                        recursive_list(entry_path)
                    else:
                        # Cast modification time and file size.
                        st_mtime = (
                            float(attr.st_mtime) if attr.st_mtime is not None else 0.0
                        )
                        st_size = int(attr.st_size) if attr.st_size is not None else 0
                        modified_iso = datetime.fromtimestamp(
                            st_mtime
                        ).isoformat()
                        size_mb = round(st_size / (1024 * 1024), 2)
                        file_info = {
                            "path": entry_path,
                            "size": size_mb,  # in MB
                            "modified": modified_iso,
                        }
                        file_list.append(file_info)

            recursive_list(remote_path)
            sftp.close()

        return file_list

    def create_bucket_if_not_exists(
        self,
        bucket_name: str,
    ) -> None:
        """Copies a file from an SFTP server to a GCS bucket."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        if bucket.exists():
            logging.info(f"Bucket '{bucket_name}' already exists.")
            return

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
        storage_client.create_bucket(bucket, location="us")


def split_csv_and_generate_signed_urls(
    bucket_name: str,
    source_blob_name: str,
    url_expiration_minutes: int = 1440
) -> Dict[str, Dict[str, str]]:
    """
    Fetches a CSV from Google Cloud Storage, splits it by a specified column, uploads the results,
    and returns a dictionary where each key is an institution ID and the value is another dictionary
    containing both a signed URL and file name for the uploaded file.

    Parameters:
        storage_client (storage.Client): The Google Cloud Storage client instance.
        bucket_name (str): The name of the GCS bucket containing the source CSV.
        source_blob_name (str): The blob name of the source CSV file.
        destination_folder (str): The destination folder in the bucket to store split files.
        institution_column (str): The column to split the CSV file on.
        url_expiration_minutes (int): The duration in minutes for which the signed URLs are valid.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary with institution IDs as keys and dictionaries with 'signed_url' and 'file_name' as values.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    source_blob = bucket.blob(source_blob_name)

    try:
        logging.info(f"Attempting to download the source blob: {source_blob_name}")
        csv_string = source_blob.download_as_text()
        csv_data = io.StringIO(csv_string)
        df = pd.read_csv(csv_data)
        logging.info("CSV data successfully loaded into DataFrame.")
    except Exception as e:
        logging.error(f"Failed to process blob {source_blob_name}: {e}")
        return {}

    pattern = re.compile(r'(?=.*institution)(?=.*id)', re.IGNORECASE)
    institution_column = None

    # Identify the correct column based on the pattern
    for column in df.columns:
        if pattern.search(column):
            institution_column = column
            logging.info(f"Matching column found: {column}")
            break
    
    if not institution_column:
        error_message = "No column found matching the pattern for 'institution' and 'id'."
        logging.error(error_message)
        return {"error": error_message}

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_data = {}

    # Processing the DataFrame
    unique_inst_ids = df[institution_column].unique()
    for inst_id in unique_inst_ids:
        group = df[df[institution_column] == inst_id]
        output = io.StringIO()
        group.to_csv(output, index=False)
        output.seek(0)

        file_name = f"{source_blob_name.split('.')[0]}_{inst_id}.csv"
        timestamped_folder = f"{inst_id}"
        new_blob_name = f"{timestamped_folder}/{file_name}"
        new_blob = storage_client.bucket(bucket_name).blob(new_blob_name)

        # Attempt to upload the CSV file
        try:
            logging.info(f"Uploading split CSV for institution ID {inst_id} to {new_blob_name}")
            new_blob.upload_from_string(output.getvalue(), content_type='text/csv')
        except Exception as e:
            logging.error(f"Failed to upload CSV for institution ID {inst_id}: {e}")
            continue
        
        # Attempt to generate a signed URL for the new blob
        try:
            expiration_time = datetime.now() + timedelta(minutes=url_expiration_minutes)
            signed_url = new_blob.generate_signed_url(expiration=expiration_time)
            all_data[inst_id] = {'signed_url': signed_url, 'file_name': file_name}
            logging.info(f"Signed URL generated successfully for institution ID {inst_id}")
        except Exception as e:
            logging.error(f"Failed to generate signed URL for institution ID {inst_id}: {e}")
            continue

    return all_data

def fetch_institution_ids(pdp_ids: list, backend_api_key: str) -> Any:
    """
    Fetches institution IDs for a list of PDP IDs using an API and returns a dictionary of valid IDs and a list of problematic IDs.

    Args:
        pdp_ids (list): List of PDP IDs to process.
        backend_api_key (str): API key required for authorization.

    Returns:
        dict: A dictionary mapping PDP ID to Institution ID for successful fetches.
        list: List of PDP IDs that were problematic or not found.
    """
    if not backend_api_key:
        raise ValueError("Missing BACKEND_API_KEY in environment variables.")
    
    # Dictionary to store successful PDP ID to Institution ID mappings
    inst_id_dict = {}
    # List to track PDP IDs that had issues
    problematic_ids = []

    # Obtain the access token
    token_response = requests.post(
        "https://dev-sst.datakind.org/api/v1/token-from-api-key",
        headers={"accept": "application/json", "X-API-KEY": backend_api_key},
    )
    if token_response.status_code != 200:
        logging.error(f"Failed to get token: {token_response.text}")
        problematic_ids.append(f"Failed to get token: {token_response.text}")
        return inst_id_dict, problematic_ids  # Return empty dict and list if no token is obtained

    access_token = token_response.json().get('access_token')
    if not access_token:
        logging.error("Access token not found in the response.")
        problematic_ids.append("Access token not found in the response.")
        return problematic_ids

    # Process each PDP ID in the list
    for pdp_id in pdp_ids:
        inst_response = requests.post(
            f"https://dev-sst.datakind.org/api/v1/institutions/pdp-id/{pdp_id}",
            headers={
                "accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Bearer {access_token}",
            },
        )

        if inst_response.status_code == 200:
            inst_data = inst_response.json()
            inst_id = inst_data.get('inst_id')
            if inst_id:
                inst_id_dict[pdp_id] = inst_id
            else:
                logging.error(f"No institution ID found for PDP ID: {pdp_id}")
                problematic_ids.append(pdp_id)
        else:
            logging.error(f"Failed to fetch institution ID for PDP ID {pdp_id}: {inst_response.text}")
            problematic_ids.append(pdp_id)

    return inst_id_dict, problematic_ids

def fetch_upload_url(file_name: str, institution_id: int, access_token: str) -> str:
    """
    Fetches an upload URL from an API for a given file and institution.

    Args:
    file_name (str): The name of the file for which the upload URL is needed.
    institution_id (int): The ID of the institution associated with the file.
    access_token (str): The Bearer token for Authorization header.

    Returns:
    str: The upload URL or an error message.
    """
    # Construct the URL with institution_id and file_name as parameters
    url = f"https://dev-sst.datakind.org/api/v1/institutions/{institution_id}/upload-url/{file_name}"

    # Set the headers including the Authorization header
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    # Make the GET request to the API
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        return response.text  # or response.json() if the response is JSON
    else:
        return f"Error fetching URL: {response.status_code} {response.text}"

def post_file_to_signed_url(file_path: str, signed_url: str) -> str:
    """
    Posts a file to a provided signed URL.

    Args:
    file_path (str): The local path to the file you want to upload.
    signed_url (str): The signed URL to which the file should be uploaded.

    Returns:
    str: The response from the server after attempting to upload.
    """
    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        # Prepare the files dictionary for uploading
        files = {'file': (file_path, file)}

        # POST the file to the signed URL
        response = requests.post(signed_url, files=files)

        # Check the response
        if response.status_code == 200:
            return "File uploaded successfully."
        else:
            return f"Failed to upload file: {response.status_code} {response.text}"