# Databricks notebook source
"""
This script generates synthetic student cohort and course data using the `faker` 
and `pdp` libraries, and uploads the generated CSV files to a Google Cloud 
Storage (GCS) bucket.  It's designed to run within a Databricks environment, 
utilizing Databricks widgets for input parameters and job task values for passing
filenames between tasks.

"""

import logging
import faker
import pandas as pd

from databricks.sdk.runtime import dbutils
from google.cloud import storage

from student_success_tool.generation import pdp

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger

# Databricks workspace identifier
DB_workspace = dbutils.widgets.get("DB_workspace")

# Input parameters from Databricks widgets
institution_id = dbutils.widgets.get("institution_id")
normalize_col_names = (
    None
    if dbutils.widgets.get("normalize_col_names") == ""
    else dbutils.widgets.get("normalize_col_names")
)
avg_num_courses_per_student = int(dbutils.widgets.get("avg_num_courses_per_student"))
num_students = int(dbutils.widgets.get("num_students"))


# Optional seed for Faker (for reproducibility)
seed = None if dbutils.widgets.get("seed") == "" else int(dbutils.widgets.get("seed"))

sst_job_id = dbutils.widgets.get("sst_job_id")

# Define the save directory, defaulting to a location within Unity Catalog volumes
if dbutils.widgets.get("save_dir") == "":
    bucket_name = f"{DB_workspace}_{institution_id}_sst_application"  # Bucket name based on institution ID
else:
    bucket_name = dbutils.widgets.get("save_dir")

logging.info(f"Save directory: {bucket_name}")
logging.info(f"normalize_col_names: {normalize_col_names}")
logging.info(f"seed: {seed}")
logging.info(f"num_students: {num_students}")


# Initialize Faker with optional seed and custom providers
faker.Faker.seed(seed)
FAKER = faker.Faker()
FAKER.add_provider(pdp.raw_cohort.Provider)
FAKER.add_provider(pdp.raw_course.Provider)

# Initialize GCS client
client = storage.Client()


bucket = client.bucket(bucket_name)

# Generate cohort records
cohort_records = [
    FAKER.raw_cohort_record(
        normalize_col_names=normalize_col_names, institution_id=institution_id
    )
    for _ in range(num_students)
]

# Generate course records (related to cohort records)
course_records = [
    FAKER.raw_course_record(cohort_record, normalize_col_names=normalize_col_names)
    for cohort_record in cohort_records
    for _ in range(
        FAKER.randomize_nb_elements(
            avg_num_courses_per_student, min=1
        )  # Random number of courses per student
    )
]

# Create Pandas DataFrames
df_cohort = pd.DataFrame(cohort_records)
df_course = pd.DataFrame(course_records)

logging.info(
    "Generated %s cohort records and %s course records",
    len(cohort_records),
    len(course_records),
)

# Construct file names
cohort_file_name = f"{institution_id}_{sst_job_id}_STUDENT_SEMESTER_AR_DEIDENTIFIED.csv"
course_file_name = f"{institution_id}_{sst_job_id}_COURSE_LEVEL_AR_DEID.csv"

# Construct full paths within the GCS bucket
cohort_full_path_name = f"validated/{cohort_file_name}"
course_full_path_name = f"validated/{course_file_name}"

# Create GCS blob objects
cohort_blob = bucket.blob(cohort_full_path_name)
course_blob = bucket.blob(course_full_path_name)

# Upload DataFrames to GCS as CSVs
cohort_blob.upload_from_string(df_cohort.to_csv(header=True, index=False), "text/csv")
course_blob.upload_from_string(df_course.to_csv(header=True, index=False), "text/csv")

print(
    f"Generated and uploaded files: {cohort_file_name} and {course_file_name} to GCS bucket: {bucket_name}"
)

# Set Databricks job task values for downstream tasks to access filenames
dbutils.jobs.taskValues.set(key="cohort_file_name", value=cohort_file_name)
dbutils.jobs.taskValues.set(key="course_file_name", value=course_file_name)