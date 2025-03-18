"""Task for publishing data that has been output by the inference task."""

import argparse
import logging
from databricks.sdk.runtime import dbutils  # noqa: F401


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
        "--sst_job_id",
        required=True,
        help="The job run id of the current pipeline execution.",
    )
    parser.add_argument(
        "--file_name_list",
        required=True,
        help="List of files separated by commas. E.g. --file_name_list=file1.csv,file2.csv",
    )
    args = parser.parse_args()
    logging.info("Deleting files from raw storage")
    for f in args.file_name_list.split(","):
        # Construct filename to delete
        fname = f"/Volumes/{args.db_workspace}/{args.institution_name}_bronze/bronze_volume/inference_jobs/{args.sst_job_id}/raw_files/{f}"
        dbutils.fs.rm(fname, recurse=False)
        # TODO: also delete the transformed tables? yes.


if __name__ == "__main__":
    main()
