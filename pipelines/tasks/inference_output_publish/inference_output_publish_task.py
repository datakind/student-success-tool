"""Task for publishing data that has been output by the inference task."""

import argparse
import logging

from ..utils.gcsutils import publish_inference_output_files
from ..utils.emails import send_inference_completion_email
from databricks.sdk import WorkspaceClient


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
    w = WorkspaceClient()
    logging.info("Publishing files to GCP bucket")
    publish_inference_output_files(
        args.db_workspace,
        args.institution_name,
        args.external_bucket_name,
        args.sst_job_id,
        False,  # Set approved = False since this task will run immediately after inference.
    )
    username = w.dbutils.secrets.get(scope="sst", key="MANDRILL_USERNAME")
    sender_email = Address("Datakind Info", "help", "datakind.org")
    cc_email_list = ["education@datakind.org"]
    password = w.dbutils.secrets.get(scope="sst", key="MANDRILL_PASSWORD")
    logging.info("Sending email notification")
    send_inference_completion_email(
        sender_email, [args.email_recipient], cc_email_list, username, password
    )


if __name__ == "__main__":
    main()
