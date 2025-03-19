"""Task for publishing data that has been output by the inference task."""

import argparse
import logging

# TODO uncomment and use the utils after pushing the fix to the repo
# from student_success_tool.pipeline_utils.gcsutils import publish_inference_output_files
# from student_success_tool.pipeline_utils.emails import send_inference_completion_email

from student_success_tool.utils.gcs import publish_inference_output_files
from student_success_tool.utils.emails import send_inference_completion_email
from databricks.sdk import WorkspaceClient
from email.headerregistry import Address

# """
# Utility functions that interact with Google Cloud Storage.
# For instance, copying files from Databricks volumes into GCP buckets.
# """

# from google.cloud import storage
# from google.cloud.storage.bucket import Bucket
# from databricks.sdk.runtime import dbutils  # noqa: F401
# import os


# def save_file(
#     bucket: Bucket, src_volume_filepath: str, dest_bucket_pathname: str
# ) -> None:
#     """Save file from databricks volume to a GCP bucket path.

#     Args:
#       src_volume_filepath: The source filepath to the Databricks volume.
#       dest_bucket_pathname: The destination filepath in GCP.

#     Returns:
#       Nothing.
#     """
#     blob = bucket.blob(dest_bucket_pathname)
#     if blob.exists():
#         raise ValueError(dest_bucket_pathname + ": File already exists in bucket.")
#     blob.upload_from_filename(src_volume_filepath)


# def publish_inference_output_files(
#     db_workspace: str,
#     institution_name: str,
#     external_bucket_name: str,
#     sst_job_id: str,
#     approved: bool,
# ) -> str:
#     """Publish output files to bucket, with folder determined based on approved parameter,
#     and return the status of the job as a string.

#     Args:
#       db_workspace: The Databricks workspace to get files from.
#       institution_name: The Databricks institution name substring used in the schema (e.g. it would be
#         'uni_of_datakind' if the gold schema was 'uni_of_datakind_gold') -- this should match the
#         "Databricksified" name generated by the webapp.
#       external_bucket_name: The destination bucket in GCP.
#       sst_job_id: the job run id of this task.
#       approved: whether this file should be published in an approved state or not.

#     Returns:
#       The status string of the job run from Databricks.
#     """
#     # Destination: The output directory in the bucket that the files get moved to.
#     bucket_directory = f"{'approved' if approved else 'unapproved'}/{sst_job_id}"
#     # Source: The Databricks volume path we want to read the files from.
#     volume_path_top_level = f"/Volumes/{db_workspace}/{institution_name}_gold/gold_volume/inference_jobs/{sst_job_id}/ext"
#     volume_path_inference_folder = f"{volume_path_top_level}/inference_output"

#     storage_client = storage.Client()
#     bucket = storage_client.bucket(external_bucket_name)

#     # Full filepaths for the Databricks volume.
#     files_to_move = []
#     status_string = ""
#     # There will be other files in the inference_output folder generated by Databricks
#     # including time started/completed, and completion status.
#     for f in dbutils.fs.ls(volume_path_inference_folder):
#         filename = f.name
#         if filename.startswith("_"):
#             if not filename.startswith("_committed") and not filename.startswith(
#                 "_started"
#             ):
#                 status_string = filename
#         elif filename.endswith(".csv"):
#             files_to_move.append(f"{volume_path_inference_folder}/{filename}")
#     # The top level volume directory can include other files that are part
#     # of the published set of files. So we still need to iterate over these files. We only take .pngs from this directory
#     # we have to be selective with the extension type because there may be subfolders in this folder.
#     for f in dbutils.fs.ls(volume_path_top_level):
#         if f.name.endswith(".png"):
#             files_to_move.append(f"{volume_path_top_level}/{f.name}")

#     for f in files_to_move:
#         save_file(
#             bucket,
#             f,
#             f"{bucket_directory}/{os.path.basename(f)}",
#         )
#     return status_string


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DB_workspace", required=True, help="Databricks workspace of the task."
    )
    parser.add_argument(
        "--databricks_institution_name",
        required=True,
        help="IThe Databricks institution name whose output we want to publish",
    )
    parser.add_argument(
        "--gcp_bucket_name",
        required=True,
        help="Name of the bucket we want to publish results to (in format [env]_[inst_id]).",
    )
    parser.add_argument(
        "--db_run_id",
        required=True,
        help="The job run id of the current pipeline execution.",
    )

    parser.add_argument(
        "--notification_email",
        required=True,
        help="User's email who triggered the inference run.",
    )
    args = parser.parse_args()
    w = WorkspaceClient()
    logging.info("Publishing files to GCP bucket")
    publish_inference_output_files(
        args.DB_workspace,
        args.databricks_institution_name,
        args.gcp_bucket_name,
        args.db_run_id,
        False,  # Set approved = False since this task will run immediately after inference.
    )
    username = w.dbutils.secrets.get(scope="sst", key="MANDRILL_USERNAME")
    sender_email = Address("Datakind Info", "help", "datakind.org")
    cc_email_list = ["education@datakind.org"]
    password = w.dbutils.secrets.get(scope="sst", key="MANDRILL_PASSWORD")
    logging.info("Sending email notification")
    send_inference_completion_email(
        sender_email, [args.notification_email], cc_email_list, username, password
    )


if __name__ == "__main__":
    main()
