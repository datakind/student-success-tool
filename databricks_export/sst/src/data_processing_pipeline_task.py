# Databricks notebook source
"""
This script prepares data for inference in the Student Success Tool (SST) pipeline.

It reads validated course and cohort data from Delta Lake tables, creates a student-term 
dataset, applies target variable logic (currently using a workaround due to library 
limitations), and saves the processed dataset back to a Delta Lake table.  It's 
designed to run within a Databricks environment.

"""

import logging

import mlflow
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
import pandas as pd

import student_success_tool.dataio as dataio
import student_success_tool.targets.pdp as targets
from student_success_tool.schemas import pdp as schemas
import student_success_tool.preprocessing.pdp as preprocessing


# Disable mlflow autologging (due to Databricks issues during feature selection)
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger

# Attempt to create a Spark session
try:
    spark_session = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("Unable to create Spark session; are you in a Databricks runtime?")
    spark_session = None

# Databricks workspace identifier
DB_workspace = dbutils.widgets.get("DB_workspace")

# Input parameters from Databricks widgets
institution_id = dbutils.widgets.get("institution_id")
sst_job_id = dbutils.widgets.get("sst_job_id")

# Delta Lake table details (read from job task values set by data ingestion task)
catalog = DB_workspace
read_schema = f"{institution_id}_bronze"
write_schema = f"{institution_id}_silver"

# Read DataFrames from Delta Lake tables (if Spark session is available)
if spark_session:
    df_course = schemas.RawPDPCourseDataSchema(
        dataio.from_delta_table(
            f"{catalog}.{read_schema}.{sst_job_id}_course_dataset_validated",
            spark_session=spark_session,
        )
    )

    df_cohort = schemas.RawPDPCohortDataSchema(
        dataio.from_delta_table(
            f"{catalog}.{read_schema}.{sst_job_id}_cohort_dataset_validated",
            spark_session=spark_session,
        )
    )
else:
    logging.warning("Spark session not initialized. Cannot read dataframes.")
    exit()  # Exit the script if the Spark session is not available.


# Reading the parameters from the institution's configuration file
cfg = dataio.read_config(
    f"/Volumes/{DB_workspace}/{institution_id}_bronze/pdp_pipeline_internal/configuration_files/{institution_id}.toml",
    schema = schemas.PDPProjectConfigV2,
)

# Read preprocessing features
min_passing_grade = cfg.preprocessing.features.min_passing_grade
min_num_credits_full_time = cfg.preprocessing.features.min_num_credits_full_time
course_level_pattern = cfg.preprocessing.features.course_level_pattern
key_course_subject_areas = cfg.preprocessing.features.key_course_subject_areas
key_course_ids = cfg.preprocessing.features.key_course_ids

# Read preprocessing target params
min_num_credits_checkin = cfg.preprocessing.target.params["min_num_credits_checkin"]
min_num_credits_target = cfg.preprocessing.target.params["min_num_credits_target"]

# Create student-term dataset
df_student_terms = preprocessing.dataops.make_student_term_dataset(
    df_cohort,
    df_course,
    min_passing_grade=min_passing_grade,
    min_num_credits_full_time=min_num_credits_full_time,
    course_level_pattern=course_level_pattern,
    key_course_subject_areas=key_course_subject_areas,
    key_course_ids=key_course_ids,
)


student_criteria = {
    "enrollment_type": ["FIRST-TIME", "RE-ADMIT", "TRANSFER-IN"],
    # "enrollment_intensity_first_term": ["FULL-TIME", "PART-TIME"], # Example, but commented out.
    # "credential_type_sought_year_1": "Associate's Degree",  # Example, but commented out.
}
intensity_time_limits = [
    ("FULL-TIME", 1.0, "year"),
    ("PART-TIME", 1.0, "year"),
]

student_id_col = "student_guid"
eligible_students = targets.shared.select_students_by_criteria(
    df_student_terms,
    student_id_cols=student_id_col,
    **student_criteria,
)
max_term_rank = df_student_terms["term_rank"].max()

df_processed = pd.merge(
    df_student_terms.loc[df_student_terms["term_rank"].eq(max_term_rank), :],
    eligible_students,
    on=student_id_col,
    how="inner",
)
df_processed = preprocessing.dataops.clean_up_labeled_dataset_cols_and_vals(df_processed)

# Save processed dataset to Delta Lake (if Spark session is available)
if spark_session:
    write_table_path = f"{catalog}.{write_schema}.{sst_job_id}_processed_dataset"
    dataio.to_delta_table(
        df_processed, write_table_path, spark_session=spark_session
    )
    logging.info(f"Processed dataset written to: {write_table_path}")
else:
    logging.warning("Spark session not initialized. Cannot write processed dataset.")

dbutils.jobs.taskValues.set(key="processed_dataset_path", value=write_table_path)