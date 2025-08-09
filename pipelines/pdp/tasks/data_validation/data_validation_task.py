"""Task for validating data using TFDV.

This task can be used to validate data against a reference schema, generate a
set of statistics from a data source and raise an exception if anomalies are
detected. It can be used to validate data in both training and inference
pipelines.
"""

import argparse
import logging
import os

from databricks.connect import DatabricksSession
import pandas as pd
import pyspark
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2


# pylint: disable=no-member
class DataValidationTask:
    """Task for validating data using TFDV."""

    def load_reference_schema(self, schema_path: str) -> schema_pb2.Schema:
        """Loads a reference schema from a pbtxt file.

        Args:
          schema_path: The path to the schema pbtxt file.

        Returns:
          The schema loaded from the pbtxt file.
        """
        logging.info("Loading reference schema from: %s", schema_path)
        return tfdv.load_schema_text(schema_path)

    def generate_statistics_from_delta_table(
        self,
        spark_session: pyspark.sql.session.SparkSession,
        input_table_path: str,
        table_format: str = "delta",
    ) -> statistics_pb2.DatasetFeatureStatisticsList:
        """Generates statistics from a delta table.

        Args:
          spark_session: The Spark session to use for reading the table.
          input_table_path: The path to the delta table to read.
          table_format: The format of the delta table.

        Returns:
          The statistics generated from the delta table.
        """
        logging.info("Generating statistics from delta table: %s", input_table_path)
        df = spark_session.read.format(table_format).table(input_table_path).toPandas()
        logging.info("Generated statistics from delta table.")
        return tfdv.generate_statistics_from_dataframe(df)

    def generate_statistics_from_csv(
        self, input_csv_path: str
    ) -> statistics_pb2.DatasetFeatureStatisticsList:
        """Generates statistics from a CSV file.

        Args:
          input_csv_path: The path to the CSV file to read.

        Returns:
          The statistics generated from the CSV file.
        """
        logging.info("Generating statistics from csv: %s", input_csv_path)
        df = pd.read_csv(input_csv_path)
        return tfdv.generate_statistics_from_dataframe(df)

    def generate_anomalies(
        self,
        stats: statistics_pb2.DatasetFeatureStatisticsList,
        schema: schema_pb2.Schema,
        environment: str,
    ) -> anomalies_pb2.Anomalies:
        """Generates anomalies from the statistics and schema.

        Args:
          stats: The statistics to validate.
          schema: The schema to validate against.
          environment: The environment to validate in (TRAINING or SERVING).

        Returns:
          The anomalies generated from the statistics and schema.
        """
        logging.info("Generating anomalies.")
        anomalies = tfdv.validate_statistics(
            statistics=stats, schema=schema, environment=environment
        )
        return anomalies

    def save_anomalies(
        self, anomalies: anomalies_pb2.Anomalies, output_path: str
    ) -> None:
        """Saves the anomalies to a pbtxt file.

        Args:
          anomalies: The anomalies to save.
          output_path: The path to save the anomalies to.

        Returns:
          None
        """
        logging.info("Saving anomalies to: %s", output_path)
        tfdv.write_anomalies_text(
            anomalies, os.path.join(output_path, "anomalies.pbtxt")
        )

    def save_statistics(
        self, statistics: statistics_pb2.DatasetFeatureStatisticsList, output_path: str
    ) -> None:
        """Saves the statistics to a pbtxt file.

        Args:
          statistics: The summary statistics to save.
          output_path: The path to save the statistics to.

        Returns:
          None
        """
        logging.info("Saving statistics to: %s", output_path)
        tfdv.write_stats_text(statistics, os.path.join(output_path, "statistics.pbtxt"))

    def check_for_anomalies(
        self,
        anomalies: anomalies_pb2.Anomalies,
        output_path: str,
        fail_on_anomalies: bool = False,
    ) -> None:
        """Checks for anomalies in the anomalies proto.

        Args:
          anomalies: The anomalies to check.
          output_path: The path to save the anomalies to.
          fail_on_anomalies: Whether to fail the job if anomalies are found.

        Returns:
          None
        """
        self.save_anomalies(anomalies, output_path)
        logging.info(
            "Checking for anomalies. fail_on_anomalies is set to %s", fail_on_anomalies
        )
        if anomalies.anomaly_info:
            logging.warning("Anomalies found. Check: %s/anomalies.pbtxt", output_path)
            if fail_on_anomalies:
                logging.error(anomalies.anomaly_info.items())
                raise ValueError("Anomalies found, pipline error.")
            logging.warning(anomalies.anomaly_info.items())
        else:
            logging.info("No anomalies found.")


def main():
    """Main function for validating data using TFDV."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_table_path", required=True, help="Path to input table to validate."
    )
    parser.add_argument(
        "--input_table_format",
        required=False,
        default="delta",
        help="Input data table format.",
    )
    parser.add_argument(
        "--input_schema_path", required=True, help="Path to input schema."
    )
    parser.add_argument(
        "--output_artifact_path", required=True, help="Path to write output artifacts."
    )
    parser.add_argument(
        "--environment",
        required=True,
        help="Environment to use for validation (TRAINING or SERVING).",
    )
    parser.add_mutually_exclusive_group(required=False)
    parser.add_argument(
        "--fail_on_anomalies_true", dest="fail_on_anomalies", action="store_true"
    )
    parser.add_argument(
        "--fail_on_anomalies_false", dest="fail_on_anomalies", action="store_false"
    )
    parser.set_defaults(fail_on_anomalies=True)
    args = parser.parse_args()

    task = DataValidationTask()

    spark_session = DatabricksSession.builder.getOrCreate()
    stats = task.generate_statistics_from_delta_table(
        spark_session, args.input_table_path, args.input_table_format
    )
    task.save_statistics(stats, args.output_artifact_path)
    schema = task.load_reference_schema(args.input_schema_path)
    anomalies = task.generate_anomalies(stats, schema, args.environment)
    task.check_for_anomalies(
        anomalies, args.output_artifact_path, fail_on_anomalies=args.fail_on_anomalies
    )


if __name__ == "__main__":
    main()
