"""
This script performs the data preprocessing step for inference in the Student Success Tool (SST) pipeline.

It reads validated course and cohort data from Delta Lake tables, creates a student-term
dataset, applies target variable logic, and saves the processed dataset back to a
Delta Lake table.

The script is designed to run within a Databricks environment as a job task, leveraging
Databricks utilities for job task values and Spark session management.


This is a POC script, it is advised to review and tests before using in production.
"""

import logging
import argparse
import mlflow
import pandas as pd
import sys
import importlib

from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils

import student_success_tool.dataio as dataio

# import student_success_tool.preprocessing.targets.pdp as targets
from student_success_tool import preprocessing
from student_success_tool.preprocessing import selection
from student_success_tool.configs.pdp import PDPProjectConfig

# Disable mlflow autologging (due to Databricks issues during feature selection)
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger


class DataProcessingTask:
    """Encapsulates the data preprocessing logic for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        """
        Initializes the DataProcessingTask.

        Args:
            args: The parsed command-line arguments.
        """
        self.spark_session = self.get_spark_session()
        self.args = args
        self.cfg = self.read_config(self.args.toml_file_path)

    def get_spark_session(self) -> DatabricksSession | None:
        """
        Attempts to create a Spark session.
        Returns:
            DatabricksSession | None: A Spark session if successful, None otherwise.
        """
        try:
            spark_session = DatabricksSession.builder.getOrCreate()
            logging.info("Spark session created successfully.")
            return spark_session
        except Exception:
            logging.error("Unable to create Spark session.")
            raise

    def read_config(self, toml_file_path: str):
        """Reads the institution's model's configuration file."""
        try:
            cfg = dataio.read_config(toml_file_path, schema=PDPProjectConfig)
            return cfg
        except FileNotFoundError:
            logging.error("Configuration file not found at %s", toml_file_path)
            raise
        except Exception as e:
            logging.error("Error reading configuration file: %e", e)
            raise

    def read_data_from_delta(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
        """
        Reads course and cohort data from Delta Lake tables.

        Returns:
            A tuple containing the course and cohort DataFrames, or (None, None) if the
            Spark session is not available.
        """
        if self.spark_session:
            try:
                df_course = schemas.raw_course.RawPDPCourseDataSchema(
                    dataio.from_delta_table(
                        self.args.course_dataset_validated_path,
                        spark_session=self.spark_session,
                    )
                )

                df_cohort = schemas.raw_cohort.RawPDPCohortDataSchema(
                    dataio.from_delta_table(
                        self.args.cohort_dataset_validated_path,
                        spark_session=self.spark_session,
                    )
                )
                return df_course, df_cohort
            except Exception as e:
                logging.error("Error reading data from Delta Lake: %s", e)
                raise
        else:
            logging.error("Spark session not initialized. Cannot read delta tables.")
            raise

    def preprocess_data(
        self, df_course: pd.DataFrame, df_cohort: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Preprocesses the data: creates student-term dataset, applies target variable logic.

        Args:
            df_course: The course DataFrame.
            df_cohort: The cohort DataFrame.
            cfg: institution's model's configuration file.

        Returns:
            The processed DataFrame.  Returns an empty DataFrame if input data is None.
        """

        # Read preprocessing features from config
        min_passing_grade = self.cfg.preprocessing.features.min_passing_grade
        min_num_credits_full_time = (
            self.cfg.preprocessing.features.min_num_credits_full_time
        )
        course_level_pattern = self.cfg.preprocessing.features.course_level_pattern
        core_terms = self.cfg.preprocessing.features.core_terms
        key_course_subject_areas = (
            self.cfg.preprocessing.features.key_course_subject_areas
        )
        key_course_ids = self.cfg.preprocessing.features.key_course_ids

        # Read preprocessing target parameters from config
        student_criteria = self.cfg.preprocessing.selection.student_criteria
        student_id_col = self.cfg.student_id_col

        # Create student-term dataset
        df_student_terms = preprocessing.pdp.make_student_term_dataset(
            df_cohort,
            df_course,
            min_passing_grade=min_passing_grade,
            min_num_credits_full_time=min_num_credits_full_time,
            course_level_pattern=course_level_pattern,
            core_terms=core_terms,
            key_course_subject_areas=key_course_subject_areas,
            key_course_ids=key_course_ids,
        )
        eligible_students = selection.pdp.select_students_by_attributes(
            df_student_terms, student_id_cols=student_id_col, **student_criteria
        )
        max_term_rank = df_student_terms["term_rank"].max()

        df_processed = pd.merge(
            df_student_terms.loc[df_student_terms["term_rank"].eq(max_term_rank), :],
            eligible_students,
            on=student_id_col,
            how="inner",
        )

        df_processed = preprocessing.pdp.clean_up_labeled_dataset_cols_and_vals(
            df_processed
        )
        logging.info("Processed dataset: %s", df_processed.shape)
        return df_processed

    def write_data_to_delta(self, df_processed: pd.DataFrame):
        """
        Saves the processed dataset to a Delta Lake table.

        Args:
            df_processed: The processed DataFrame.
        """
        if not self.spark_session:
            logging.error(
                "Spark session not initialized. Cannot write processed dataset."
            )
            return

        write_schema = f"{self.args.databricks_institution_name}_silver"
        write_table_path = f"{self.args.DB_workspace}.{write_schema}.{self.args.db_run_id}_processed_dataset"

        try:
            dataio.to_delta_table(
                df_processed, write_table_path, spark_session=self.spark_session
            )
            logging.info("Processed dataset written to table: %s", write_table_path)
            return write_table_path

        except Exception as e:
            logging.error("Error writing processed data to Delta Lake: %s", e)
            raise

    def run(self):
        """Executes the data preprocessing pipeline."""
        df_course, df_cohort = self.read_data_from_delta()
        df_processed = self.preprocess_data(df_course, df_cohort)
        processed_dataset_path = self.write_data_to_delta(df_processed)

        # Setting task variables for downstream tasks
        dbutils.jobs.taskValues.set(
            key="processed_dataset_path", value=processed_dataset_path
        )
        dbutils.jobs.taskValues.set(
            key="toml_file_path", value=self.args.toml_file_path
        )


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data preprocessing for inference in the SST pipeline."
    )
    parser.add_argument(
        "--DB_workspace",
        type=str,
        required=True,
        help="Databricks workspace identifier",
    )
    parser.add_argument(
        "--databricks_institution_name",
        type=str,
        required=True,
        help="Databricks institution name",
    )
    parser.add_argument(
        "--db_run_id", type=str, required=True, help="Databricks run ID"
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--cohort_dataset_validated_path",
        type=str,
        required=True,
        help="Path to delta table containing cohort dataset",
    )
    parser.add_argument(
        "--course_dataset_validated_path",
        type=str,
        required=True,
        help="Path to delta table containing course dataset",
    )
    parser.add_argument(
        "--toml_file_path", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--custom_schemas_path",
        required=False,
        help="Folder path to store custom schemas folders",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        sys.path.append(args.custom_schemas_path)
        sys.path.append(
            f"/Volumes/staging_sst_01/{args.databricks_institution_name}_bronze/bronze_volume/inference_inputs"
        )
        schemas = importlib.import_module("schemas")
        # schemas = importlib.import_module(f"{args.databricks_institution_name}.schemas")
        logging.info("Running task with custom schema")
    except Exception:
        from student_success_tool.dataio.schemas import pdp as schemas

        logging.info("Running task with default schema")
    task = DataProcessingTask(args)
    task.run()
