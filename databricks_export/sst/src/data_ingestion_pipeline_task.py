# Databricks notebook source
"""
This script ingests course and cohort data for the Student Success Tool (SST) pipeline.

It reads data from CSV files stored in a Google Cloud Storage (GCS) bucket, 
performs schema validation using the `pdp` library, and writes the validated data 
to Delta Lake tables in Databricks Unity Catalog.

The script is designed to run within a Databricks environment, leveraging Databricks 
utilities for widget input, job task values, and Spark session management. It also 
handles cases where a Spark session cannot be initialized (e.g., running locally).

"""

import logging
import os

from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
from google.cloud import storage

import student_success_tool.dataio as dataio
from student_success_tool.schemas import pdp as schemas

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger

# Attempt to create a Spark session. Handles exceptions if not in Databricks.
try:
    spark_session = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("Unable to create Spark session; are you in a Databricks runtime?")
    spark_session = None

# Input parameters (provided via Databricks widgets or job task values)
DB_workspace = dbutils.widgets.get("DB_workspace")  # Databricks workspace identifier

institution_id = dbutils.widgets.get("institution_id")
sst_job_id = dbutils.widgets.get("sst_job_id")

# Handle synthetic data generation case
if dbutils.widgets.get("synthetic_needed") == "True":
    course_file_name = dbutils.jobs.taskValues.get(
        taskKey="generate_synthetic_data", key="course_file_name"
    )
    cohort_file_name = dbutils.jobs.taskValues.get(
        taskKey="generate_synthetic_data", key="cohort_file_name"
    )
else:
    course_file_name = dbutils.widgets.get("course_file_name")
    cohort_file_name = dbutils.widgets.get("cohort_file_name")


# Define paths (using Unity Catalog volumes)
internal_pipeline_path = f"/Volumes/{DB_workspace}/{institution_id}_bronze/pdp_pipeline_internal/{sst_job_id}/raw_files/"


# Create internal pipeline directory
os.makedirs(internal_pipeline_path, exist_ok=True)

# Initialize GCS client
storage_client = storage.Client()
bucket_name = f"{DB_workspace}_{institution_id}_sst_application"

bucket = storage_client.bucket(bucket_name)
sst_container_folder = "validated"

# Download course data from GCS
course_blob_name = f"{sst_container_folder}/{course_file_name}"
course_blob = bucket.blob(course_blob_name)
course_blob.download_to_filename(f"{internal_pipeline_path}{course_file_name}")

# Download cohort data from GCS
cohort_blob_name = f"{sst_container_folder}/{cohort_file_name}"
cohort_blob = bucket.blob(cohort_blob_name)
cohort_blob.download_to_filename(f"{internal_pipeline_path}{cohort_file_name}")


# Set path_volume (important for compatibility with Datakind's code)
path_volume = internal_pipeline_path

# Construct full file paths
fpath_course = os.path.join(path_volume, course_file_name)
fpath_cohort = os.path.join(path_volume, cohort_file_name)

# Read data from CSV files into Pandas DataFrames and validate schema
df_course = dataio.pdp.read_raw_course_data(
    file_path=fpath_course, schema=schemas.RawPDPCourseDataSchema, dttm_format="%Y-%m-%d"
)
df_cohort = dataio.pdp.read_raw_cohort_data(
    file_path=fpath_cohort, schema=schemas.RawPDPCohortDataSchema
)


# Define Delta Lake table details
catalog = DB_workspace
write_schema = f"{institution_id}_bronze"


# Write DataFrames to Delta Lake tables (only if Spark session is available)
if spark_session:
    dataio.to_delta_table(
        df_course,
        f"{catalog}.{write_schema}.{sst_job_id}_course_dataset_validated",
        spark_session=spark_session,
    )

    dataio.to_delta_table(
        df_cohort,
        f"{catalog}.{write_schema}.{sst_job_id}_cohort_dataset_validated",
        spark_session=spark_session,
    )

    # Verify Delta Lake write by reading data back
    df_course_from_catalog = schemas.RawPDPCourseDataSchema(
        dataio.from_delta_table(
            f"{catalog}.{write_schema}.{sst_job_id}_course_dataset_validated",
            spark_session=spark_session,
        )
    )
    print(f"Course DataFrame shape from catalog: {df_course_from_catalog.shape}")

    df_cohort_from_catalog = schemas.RawPDPCohortDataSchema(
        dataio.from_delta_table(
            f"{catalog}.{write_schema}.{sst_job_id}_cohort_dataset_validated",
            spark_session=spark_session,
        )
    )
    print(f"Cohort DataFrame shape from catalog: {df_cohort_from_catalog.shape}")
else:
    logging.warning(
        "Spark session not initialized. Skipping Delta Lake write and verification."
    )