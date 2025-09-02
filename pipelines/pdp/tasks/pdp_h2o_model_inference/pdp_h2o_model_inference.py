"""
This script performs model inference for the Student Success Tool (SST) pipeline.

It loads a pre-trained ML model from MLflow Model run,
reads a processed dataset from Delta Lake, performs inference, calculates SHAP values,
and writes the predictions back to Delta Lake.

The script is designed to run within a Databricks environment as a job task, leveraging
Databricks utilities for job task values and Spark session management.

This is a POC script, it is advised to review and tests before using in production.
"""

# Import necessary libraries
import logging
import os
import argparse
import typing as t
import sys
import importlib

import h2o
import mlflow
import numpy as np
import pandas as pd
import shap
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from email.headerregistry import Address
import numpy.typing as npt


# Import project-specific modules
import student_success_tool.dataio as dataio
from student_success_tool import modeling as modeling
from student_success_tool.modeling import inference
import pkgutil
print("configs at:", modeling.__file__)
print("submodules:", [m.name for m in pkgutil.iter_modules(modeling.__path__)])
from student_success_tool.modeling.h2o_modeling import utils as h2o_utils
from student_success_tool.modeling.h2o_modeling import inference as h2o_inference
from student_success_tool.modeling.h2o_modeling import evaluation as h2o_evaluation
from student_success_tool.modeling.h2o_modeling import imputation as h2o_imputation
import student_success_tool.configs as configs
print("configs at:", configs.__file__)
print("submodules:", [m.name for m in pkgutil.iter_modules(configs.__path__)])
from student_success_tool.configs.h2o_configs.pdp import PDPProjectConfig

from student_success_tool.modeling.evaluation import plot_shap_beeswarm
from student_success_tool.utils import emails
from mlflow.tracking import MlflowClient

# Disable mlflow autologging (prevents conflicts in Databricks environments)
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Suppress py4j logging


class ModelInferenceTask:
    """Encapsulates the model inference logic for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        """Initializes the ModelInferenceTask."""
        self.args = args
        self.spark_session = self.get_spark_session()
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

    def load_mlflow_model(self):
        client = MlflowClient(registry_uri="databricks-uc")
        full_model_name = f"{self.args.DB_workspace}.{self.args.databricks_institution_name}_gold.{self.args.model_name}"

        try:
            # Choose max version and grab associated model run id
            mv = max(
                client.search_model_versions(f"name='{full_model_name}'"),
                key=lambda v: int(v.version),
            )
            self.model_run_id = mv.run_id

            # Look up the run details and assign the experiment id
            run = client.get_run(self.model_run_id)
            self.model_experiment_id = run.info.experiment_id

            # Load h2o model
            model = h2o_utils.load_h2o_model(run_id=self.model_run_id)

            logging.info(
                "Loaded H2O model from run_id=%s (version=%s)",
                self.model_run_id,
                mv.version,
            )
            return model
        except Exception as e:
            logging.error("Error loading MLflow model via run_id: %s", e)
            raise

    def predict(
        self, model, df: pd.DataFrame, model_feature_names: t.List
    ) -> pd.DataFrame:
        """Performs inference and adds predictions to the DataFrame."""
    
        # Convert to h2o frame and run prediction
        df_predicted = df.copy()
        labels, probs = h2o_inference.predict_h2o(
            df_predicted,
            model=model,
            feature_names=model_feature_names,
            pos_label=self.cfg.pos_label,
        )

        return df_predicted.assign(
            predicted_prob=probs,
            predicted_label=labels,
        )

    def write_data_to_delta(self, df: pd.DataFrame, table_name_suffix: str):
        """Writes a DataFrame to a Delta Lake table."""
        write_schema = f"{self.args.databricks_institution_name}_silver"
        table_path = f"{self.args.DB_workspace}.{write_schema}.{table_name_suffix}"

        try:
            dataio.to_delta_table(df, table_path, spark_session=self.spark_session)
            logging.info(
                "%s data written to: %s", table_name_suffix.capitalize(), table_path
            )
        except Exception as e:
            logging.error(
                "Error writing %s data to Delta Lake: %s", table_name_suffix, e
            )
            raise

    def calculate_shap_values(
        self,
        model,
        df_processed: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """Calculates SHAP values."""

        try:
            # Load and preprocess training data
            df_train = h2o_evaluation.extract_training_data_from_model(
                automl_experiment_id=self.model_experiment_id,
            )

            train_features = h2o_imputation.SklearnImputerWrapper.load_and_transform(
                df=df_train,
                run_id=self.model_run_id,
            )

            # Sample background data for performance optimization
            bd =  train_features.sample(
                n=min(self.cfg.inference.background_data_sample, len(df_processed)),
                random_state=self.cfg.random_state,
            )

            contribs_df = h2o_inference.compute_h2o_shap_contributions(
                model=model,
                df=df_processed,
                background_data=bd,
            )
            return contribs_df
        except Exception as e:
            logging.error("Error during SHAP value calculation: %s", e)
            raise

    def top_n_features(
        self,
        grouped_features: pd.DataFrame,
        unique_ids: pd.Series,
        grouped_shap_values: npt.NDArray[np.float64],
        n: int = 10,
    ) -> pd.DataFrame:
        features_table = dataio.read_features_table("assets/pdp/features_table.toml")
        try:
            top_n_shap_features = inference.top_shap_features(
                grouped_features,
                unique_ids,
                grouped_shap_values,
                n,
                features_table=features_table,
            )
            return top_n_shap_features

        except Exception as e:
            logging.error("Error computing top %d shap features table: %s", n, e)
            return None

    def support_score_distribution(
        self,
        grouped_features,
        unique_ids,
        df_predicted,
        grouped_shap_values,
        model_feature_names,
    ):
        """
        Selects top features to display and store
        """
        if not self.spark_session:
            logging.error(
                "Spark session not initialized. Cannot post process shap values."
            )
            return None

        # --- Load features table ---
        features_table = dataio.read_features_table("assets/pdp/features_table.toml")

        # --- Inference Parameters ---
        inference_params = {
            "num_top_features": 5,
            "min_prob_pos_label": 0.5,
        }

        pred_probs = df_predicted["predicted_prob"]
        # --- Feature Selection for Display ---

        try:
            result = inference.support_score_distribution_table(
                grouped_features,
                unique_ids,
                pred_probs,
                grouped_shap_values,
                inference_params=inference_params,
                features_table=features_table,
                model_feature_names=model_feature_names,
            )

            return result

        except Exception as e:
            logging.error("Error computing support score distribution table: %s", e)
            return None

    def inference_shap_feature_importance(self, grouped_features, grouped_shap_values):
        """
        Selects top important features to display and store
        """
        if not self.spark_session:
            logging.error(
                "Spark session not initialized. Cannot post process shap values."
            )
            return None
        features_table = dataio.read_features_table("assets/pdp/features_table.toml")
        shap_feature_importance = inference.generate_ranked_feature_table(
            grouped_features, grouped_shap_values.values, features_table
        )

        return shap_feature_importance

    def get_top_features_for_display(
        self,
        grouped_features,
        unique_ids,
        df_predicted,
        grouped_shap_values,
    ):
        """
        Selects top features to display and store
        """
        if not self.spark_session:
            logging.error(
                "Spark session not initialized. Cannot post process shap values."
            )
            return None

        # --- Load features table ---
        features_table = dataio.read_features_table("assets/pdp/features_table.toml")

        # --- Feature Selection for Display ---
        try:
            result = inference.select_top_features_for_display(
                grouped_features,
                unique_ids,
                df_predicted["predicted_prob"],
                grouped_shap_values.values,
                n_features=self.cfg.inference.num_top_features,
                features_table=features_table,
                needs_support_threshold_prob=self.cfg.inference.min_prob_pos_label,
            )
            return result

        except Exception as e:
            logging.error("Error top features to display: %s", e)
            return None

    def run(self):
        """Executes the model inference pipeline."""
        df_processed = dataio.from_delta_table(
            self.args.processed_dataset_path, spark_session=self.spark_session
        )
        model = self.load_mlflow_model()

        # Load and transform using sklearn imputer
        df_processed = h2o_imputation.SklearnImputerWrapper.load_and_transform(
            df=df_processed,
            run_id=self.model_run_id,
        )
        
        model_feature_names = h2o_inference.get_h2o_used_features(model)
        df_features = df_processed.loc[:, model_feature_names]
        unique_ids = df_processed[self.cfg.student_id_col]

        # --- Email notify users ---
        # Uncomment below once we want to enable CC'ing to DK's email.
        # Secrets from Databricks
        w = WorkspaceClient()
        MANDRILL_USERNAME = w.dbutils.secrets.get(scope="sst", key="MANDRILL_USERNAME")
        MANDRILL_PASSWORD = w.dbutils.secrets.get(scope="sst", key="MANDRILL_PASSWORD")
        SENDER_EMAIL = Address("Datakind Info", "help", "datakind.org")
        emails.send_inference_kickoff_email(
            SENDER_EMAIL,
            [self.args.notification_email],
            [self.args.DK_CC_EMAIL],
            MANDRILL_USERNAME,
            MANDRILL_PASSWORD,
        )

        df_predicted = self.predict(model, df_features, model_feature_names)
        self.write_data_to_delta(df_predicted, "predicted_dataset")

        # --- SHAP Values Calculation ---
        shap_values = self.calculate_shap_values(model, df_features)

        if shap_values is not None:  # Proceed only if SHAP values were calculated
            logging.info(f"now cfg.model.run_id = {self.cfg.model.run_id}")
            logging.info(
                f"now cfg.model.experiment_id = {self.cfg.model.experiment_id}"
            )

            # Group shap values and features by base name
            grouped_shap_values = h2o_inference.group_shap_values(shap_values)
            grouped_features = h2o_inference.group_feature_values(df_features)

            with mlflow.start_run(run_id=self.cfg.model.run_id):
                # full_model_name = f"{self.args.DB_workspace}.{self.args.databricks_institution_name}_gold.{self.args.model_name}"
                # --- SHAP Summary Plot ---
                # shap_fig = plot_shap_beeswarm(grouped_shap_values)

                # Inference_features_with_most_impact TABLE
                inference_features_with_most_impact = self.top_n_features(
                    grouped_features, unique_ids, grouped_shap_values
                )
                support_scores = pd.DataFrame(
                    {
                        "student_id": unique_ids.values,  # From the original df_test
                        "support_score": df_predicted["predicted_prob"].values,
                    }
                )
                inference_features_with_most_impact = (
                    inference_features_with_most_impact.merge(
                        support_scores, on="student_id", how="left"
                    )
                )

                # print or log the inference_features_with_most_impact
                logging.info(
                    "Inference features with most impact:\n%s",
                    inference_features_with_most_impact,
                )
                # shap_feature_importance TABLE
                shap_feature_importance = self.inference_shap_feature_importance(
                    grouped_features, grouped_shap_values
                )
                # # support_overview TABLE
                support_overview_table = self.support_score_distribution(
                    grouped_features,
                    unique_ids,
                    df_predicted,
                    grouped_shap_values,
                    model_feature_names,
                )
                if inference_features_with_most_impact is None:
                    msg = "Inference features with most impact is empty: cannot write inference summary tables."
                    logging.error(msg)
                    raise Exception(msg)
                if shap_feature_importance is None:
                    msg = "Shap Feature Importance is empty: cannot write inference summary tables."
                    logging.error(msg)
                    raise Exception(msg)
                if support_overview_table is None:
                    msg = "Support overview table is empty: cannot write inference summary tables."
                    logging.error(msg)
                    raise Exception(msg)
                self.write_data_to_delta(
                    inference_features_with_most_impact,
                    f"inference_{self.cfg.model.run_id}_features_with_most_impact",
                )
                self.write_data_to_delta(
                    shap_feature_importance,
                    f"inference_{self.cfg.model.run_id}_shap_feature_importance",
                )
                self.write_data_to_delta(
                    support_overview_table,
                    f"inference_{self.cfg.model.run_id}_support_overview",
                )

                # Shap Result Table
                shap_results = self.get_top_features_for_display(
                    grouped_features,
                    unique_ids,
                    df_predicted,
                    grouped_shap_values,
                    model_feature_names,
                )

                # --- Save Results to ext/ folder in Gold volume. ---
                if shap_results is not None:
                    # Specify the folder for the output files to be stored.
                    result_path = f"{self.args.job_root_dir}/ext/"
                    os.makedirs(result_path, exist_ok=True)
                    print("result_path:", result_path)

                    # TODO What is the proper name for the table with the results?
                    # Write the DataFrame to Unity Catalog table
                    self.write_data_to_delta(shap_results, "inference_output")

                    # Write the DataFrame to CSV in the specified volume
                    spark_df = self.spark_session.createDataFrame(shap_results)
                    spark_df.coalesce(1).write.format("csv").option(
                        "header", "true"
                    ).mode("overwrite").save(result_path + "inference_output")
                    # Write the SHAP chart png to the volume
                    shap_fig.savefig(
                        result_path + "shap_chart.png", bbox_inches="tight"
                    )
                else:
                    logging.error(
                        "Empty Shap results, cannot create the SHAP chart and table"
                    )
                    raise Exception(
                        "Empty Shap results, cannot create the SHAP chart and table"
                    )


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform model inference for the SST pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
    parser.add_argument("--model_type", type=str, required=True, help="Model type")
    parser.add_argument(
        "--job_root_dir",
        type=str,
        required=True,
        help="Folder path to store job output files",
    )
    parser.add_argument(
        "--toml_file_path", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--processed_dataset_path",
        type=str,
        required=True,
        help="Path to processed dataset table",
    )
    parser.add_argument(
        "--notification_email",
        type=str,
        required=True,
        help="Insitution's email used for notifications",
    )
    parser.add_argument(
        "--DK_CC_EMAIL", type=str, required=True, help="Datakind email address CC'd"
    )
    parser.add_argument(
        "--modeling_table_path",
        type=str,
        required=True,
        help="Path to training dataset table used to calculate shap values",
    )
    parser.add_argument(
        "--custom_schemas_path",
        type=str,
        required=False,
        help="Folder path to store custom schemas folders",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        sys.path.append(args.custom_schemas_path)
        schemas = importlib.import_module(f"{args.databricks_institution_name}.schemas")
        logging.info("Running task with custom schema")
    except Exception:
        print("Running task with default schema")
        logging.info("Running task with default schema")
    task = ModelInferenceTask(args)
    task.run()
