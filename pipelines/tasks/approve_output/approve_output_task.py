"""Task for publishing data that has been approved and deleting the unapproved outputs for that job.

NOTE: this job does NOT move the existing bucket file from unapproved/ to approved/. It instead writes
the current files in the Databricks volume to approved/ and deletes the unapproved/ output. Which means,
if the actual file in the Databricks volume gets updated, the webapp will pick up the updates. And the
old output will be DELETED.
"""

import argparse
import logging
from google.cloud import storage

from student_success_tool.utils.gcs import publish_inference_output_files
from student_success_tool.utils.emails import send_email
from databricks.sdk import WorkspaceClient
from email.headerregistry import Address

APPROVAL_SUBJECT = "Student Success Tool: Inference Output Reviewed By Datakind."
APPROVAL_MESSAGE = """\
    Hello!

    Your Datakind Student Success Tool inference output has been reviewed by Datakind.
    """


def main():
    """Main function."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db_workspace", required=True, help="Databricks workspace of the task."
    )
    parser.add_argument(
        "--institution_name",
        required=True,
        help="IThe Databricks institution name whose output we want to publish",
    )
    parser.add_argument(
        "--external_bucket_name",
        required=True,
        help="Name of the bucket we want to publish results to (in format [env]_[inst_id]).",
    )
    parser.add_argument(
        "--sst_job_id",
        required=True,
        help="The job run id of the current pipeline execution.",
    )
    parser.add_argument(
        "--email_recipient",
        required=True,
        help="User's email who triggered the inference run.",
    )
    args = parser.parse_args()
    logging.info("Publishing files as approved to GCP bucket")
    publish_inference_output_files(
        args.db_workspace,
        args.institution_name,
        args.external_bucket_name,
        args.sst_job_id,
        True,  # Set approved = True since this task runs after human approval.
    )
    logging.info("Delete unapproved files for this job from GCP bucket.")
    w = WorkspaceClient()
    unapproved_path = f"unapproved/{args.sst_job_id}/"
    storage_client = storage.Client()
    bucket = storage_client.bucket(args.external_bucket_name)
    blobs = bucket.list_blobs(prefix=unapproved_path)
    for blob in blobs:
        blob.delete()
    username = w.dbutils.secrets.get(scope="sst", key="MANDRILL_USERNAME")
    sender_email = Address("Datakind Info", "help", "datakind.org")
    cc_email_list = ["education@datakind.org"]
    password = w.dbutils.secrets.get(scope="sst", key="MANDRILL_PASSWORD")
    logging.info("Sending email notification")
    send_email(
        sender_email,
        [args.email_recipient],
        cc_email_list,
        APPROVAL_SUBJECT,
        APPROVAL_MESSAGE,
        username,
        password,
    )


if __name__ == "__main__":
    main()
