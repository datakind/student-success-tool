"""Tests for data_validation_task."""

import os
import shutil
import unittest

import pandas as pd
import pyspark
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_metadata.proto.v0 import statistics_pb2

from data_validation_task import DataValidationTask


# pylint: disable=no-member
class TestDataValidationTask(unittest.TestCase):
    """Tests for data_validation_task."""

    @classmethod
    def setUpClass(cls):
        """Set up test data and SparkSession."""
        super().setUpClass()
        cls.task = DataValidationTask()

        # Create test CSV data
        cls.test_csv_path = "test_data.csv"
        cls.test_data = pd.DataFrame(
            {
                "col1": [1, 1, 1, 1, 1],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        cls.test_data.to_csv(cls.test_csv_path, index=False)

        # Create anomalous CSV data
        cls.anomalous_csv_path = "anomalous_data.csv"
        cls.anomalous_data = pd.DataFrame(
            {
                "col1": [0, 0, 0, 0, 0],
                "col2": ["out", "of", "schema", "validation", "error"],
                "col3": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            }
        )
        cls.anomalous_data.to_csv(cls.anomalous_csv_path, index=False)

        # Create test schema
        cls.test_schema_path = "test_schema.pbtxt"
        test_stats = tfdv.generate_statistics_from_dataframe(cls.test_data)
        test_schema = tfdv.infer_schema(test_stats)
        test_schema.default_environment.append("TRAINING")
        tfdv.write_schema_text(test_schema, cls.test_schema_path)

        # Create anomalous schema
        cls.anomalous_schema_path = "anomalous_schema.pbtxt"
        anomalous_stats = tfdv.generate_statistics_from_dataframe(cls.anomalous_data)
        anomalous_schema = tfdv.infer_schema(anomalous_stats)
        anomalous_schema.default_environment.append("TRAINING")
        tfdv.write_schema_text(anomalous_schema, cls.anomalous_schema_path)

        # Create output directory
        cls.output_dir = "test_output"
        os.makedirs(cls.output_dir, exist_ok=True)

        # Create SparkSession
        cls.spark = (
            pyspark.sql.SparkSession.builder.master("local[*]")
            .appName("TestDataValidationTask")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up test files and SparkSession."""
        super().tearDownClass()
        os.remove(cls.test_csv_path)
        os.remove(cls.anomalous_csv_path)
        os.remove(cls.test_schema_path)
        os.remove(cls.anomalous_schema_path)
        shutil.rmtree(cls.output_dir)
        cls.spark.stop()

    def test_load_reference_schema(self):
        """Test loading a reference schema."""
        schema = self.task.load_reference_schema(self.test_schema_path)
        self.assertIsInstance(schema, schema_pb2.Schema)

    def test_generate_statistics_from_csv(self):
        """Test generating statistics from a CSV file."""
        stats = self.task.generate_statistics_from_csv(self.test_csv_path)
        self.assertIsInstance(stats, statistics_pb2.DatasetFeatureStatisticsList)

    def test_generate_anomalies(self):
        """Test generating anomalies."""
        stats = self.task.generate_statistics_from_csv(self.anomalous_csv_path)
        schema = self.task.load_reference_schema(self.test_schema_path)
        anomalies = self.task.generate_anomalies(stats, schema, "TRAINING")
        self.assertIsInstance(anomalies, anomalies_pb2.Anomalies)
        self.assertTrue(anomalies.anomaly_info)

    def test_save_anomalies(self):
        """Test saving anomalies."""
        stats = self.task.generate_statistics_from_csv(self.anomalous_csv_path)
        schema = self.task.load_reference_schema(self.test_schema_path)
        anomalies = self.task.generate_anomalies(stats, schema, "TRAINING")
        self.task.save_anomalies(anomalies, self.output_dir)
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "anomalies.pbtxt"))
        )

    def test_save_statistics(self):
        """Test saving statistics."""
        stats = self.task.generate_statistics_from_csv(self.anomalous_csv_path)
        self.task.save_statistics(stats, self.output_dir)
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "statistics.pbtxt"))
        )

    def test_check_for_anomalies_no_fail(self):
        """Test checking for anomalies without failing."""
        stats = self.task.generate_statistics_from_csv(self.test_csv_path)
        schema = self.task.load_reference_schema(self.test_schema_path)
        anomalies = self.task.generate_anomalies(stats, schema, "TRAINING")
        self.task.check_for_anomalies(anomalies, self.output_dir)
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "anomalies.pbtxt"))
        )

    def test_check_for_anomalies_fail(self):
        """Test checking for anomalies and failing."""
        stats = self.task.generate_statistics_from_csv(self.anomalous_csv_path)
        schema = self.task.load_reference_schema(self.test_schema_path)
        anomalies = self.task.generate_anomalies(stats, schema, "TRAINING")
        with self.assertRaises(ValueError):
            self.task.check_for_anomalies(
                anomalies, self.output_dir, fail_on_anomalies=True
            )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "anomalies.pbtxt"))
        )

    def test_check_for_anomalies_on_false(self):
        """Test checking for anomalies and override with fail_on_anomalies."""
        stats = self.task.generate_statistics_from_csv(self.anomalous_csv_path)
        schema = self.task.load_reference_schema(self.test_schema_path)
        anomalies = self.task.generate_anomalies(stats, schema, "TRAINING")
        self.task.check_for_anomalies(
            anomalies, self.output_dir, fail_on_anomalies=False
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "anomalies.pbtxt"))
        )

    def test_generate_statistics_from_delta_table(self):
        """Test generating statistics from a delta table."""
        test_spark_df = self.spark.createDataFrame(self.test_data)
        test_table_path = "test_table"
        test_spark_df.write.format("delta").saveAsTable(test_table_path)

        stats = self.task.generate_statistics_from_delta_table(
            self.spark, test_table_path
        )
        self.assertIsInstance(stats, statistics_pb2.DatasetFeatureStatisticsList)

        self.spark.sql(f"DROP TABLE {test_table_path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
