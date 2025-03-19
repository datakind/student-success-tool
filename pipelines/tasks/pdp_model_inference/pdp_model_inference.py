"""
This script performs model inference for the Student Success Tool (SST) pipeline.

It loads a pre-trained ML model from MLflow Model Registry,
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
from joblib import Parallel, delayed
from typing import List, Optional
import sys
import importlib

import functools as ft
import mlflow
import numpy as np
import pandas as pd
import shap
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from email.headerregistry import Address


# Import project-specific modules
import student_success_tool.dataio as dataio
from student_success_tool.modeling import inference
import student_success_tool.modeling as modeling
from student_success_tool.schemas.pdp import PDPProjectConfig

import sklearn as sk
print(sk.__version__)

# Disable mlflow autologging (prevents conflicts in Databricks environments)
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Suppress py4j logging

# Import that works on develop branch
# from student_success_tool.modeling.evaluation import plot_shap_beeswarm
# from student_success_tool.utils import emails


# Still using the old locations since we are installing package from branch
# from student_success_tool.pipeline_utils.plot import plot_shap_beeswarm
# from student_success_tool.pipeline_utils import emails

"""Utility functions for sending emails in Databricks."""

import smtplib

from email.message import EmailMessage

SMTP_SERVER = "smtp.mandrillapp.com"
# TODO: switch port?
SMTP_PORT = 587  # or 465 for SSL
COMPLETION_SUCCESS_SUBJECT = "Student Success Tool: Inference Results Available."
COMPLETION_SUCCESS_MESSAGE = """\
    Hello!

    Your Datakind Student Success Tool inference run has successfully completed and the results are ready for viewing on the Website.
    """

INFERENCE_KICKOFF_SUBJECT = "Student Success Tool: Inference Run In Progress."
INFERENCE_KICKOFF_MESSAGE = """\
    Hello!

    Your Datakind Student Success Tool inference run has been triggered. Once results have been checked and are ready for viewing, you'll receive another email.
    """


def send_email(
    sender_email: str,
    receiver_email_list: list[str],
    cc_email_list: list[str],
    subject: str,
    body: str,
    mandrill_username: str,
    mandrill_password: str,
) -> None:
    """Send email.

    Args:
      sender_email: Email of sender.
      receiver_email_list: List of emails to send to.
      cc_email_list: List of emails to CC.
      subject: Subject of email.
      body: Body of email.
      mandrill_username: Mandrill username for the SMTP server.
      mandrill_password: Mandrill password for the SMTP server.

    Returns:
      Nothing.
    """
    # Create the email message
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email_list
    msg["Cc"] = cc_email_list
    msg.set_content(body)
    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.ehlo()
        server.starttls()
        server.login(mandrill_username, mandrill_password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        raise e


def send_inference_completion_email(
    sender_email: str,
    receiver_email_list: list[str],
    cc_email_list: list[str],
    username: str,
    password: str,
) -> None:
    """Send email with completion of inference run message.

    Args:
      sender_email: Email of sender.
      receiver_email_list: List of emails to send to.
      cc_email_list: List of emails to CC.
      username: Mandrill username for the SMTP server.
      password: Mandrill password for the SMTP server.

    Returns:
      Nothing.
    """
    send_email(
        sender_email,
        receiver_email_list,
        cc_email_list,
        COMPLETION_SUCCESS_SUBJECT,
        COMPLETION_SUCCESS_MESSAGE,
        username,
        password,
    )


def send_inference_kickoff_email(
    sender_email: str,
    receiver_email_list: list[str],
    cc_email_list: list[str],
    username: str,
    password: str,
) -> None:
    """Send email with kickoff of inference run message.

    Args:
      sender_email: Email of sender.
      receiver_email_list: List of emails to send to.
      cc_email_list: List of emails to CC.
      username: Mandrill username for the SMTP server.
      password: Mandrill password for the SMTP server.

    Returns:
      Nothing.
    """
    send_email(
        sender_email,
        receiver_email_list,
        cc_email_list,
        INFERENCE_KICKOFF_SUBJECT,
        INFERENCE_KICKOFF_MESSAGE,
        username,
        password,
    )


import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.pyplot as plt

def plot_shap_beeswarm(shap_values):
    # Define the color palette
    start_color = (0.34, 0.79, 0.55)  # Green
    middle_color = (0.98, 0.82, 0.22)  # Yellow
    end_color = (0.95, 0.60, 0.19)  # Orange
    colors = [start_color, middle_color, end_color]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, 256)

    plt.figure(figsize=(5, 6))  # Adjust height as needed.

    # Plot the beeswarm
    ax = shap.plots.beeswarm(
        shap_values,
        show=False,
        max_display=25,
        color=cmap,
        color_bar=False,
        plot_size=(5, 25),
    )

    # Get the current y-axis tick labels (feature names)
    y_ticks_labels = [label.get_text() for label in ax.get_yticklabels()]

    # Modify the feature names
    modified_labels = [label.replace("_", " ").title() for label in y_ticks_labels]

    # Set the modified labels back to the y-axis
    ax.set_yticklabels(modified_labels)

    # Add horizontal lines
    # Get the y-axis tick positions
    y_ticks = ax.get_yticks()
    for y in y_ticks:
        # Add a dashed gray line
        ax.axhline(y=y, color="gray", linestyle="-", linewidth=0.5)

    # Add colorbar above chart
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(
            cmap=cmap,
        ),
        ax=ax,
        location="top",
        shrink=0.5,
        aspect=20,
        ticks=[],
        pad=0.02,
    )
    cbar.outline.set_visible(False)
    cbar.set_label("Feature Value", loc="center", labelpad=10)
    # Add labels to the left and right of the colorbar
    cbar.ax.text(
        -0.1,  # Adjust horizontal position for left label
        0.5,  # Vertical position (middle of colorbar)
        "Low",  # Left label text
        va="center",
        ha="right",
        transform=cbar.ax.transAxes,
    )
    cbar.ax.text(
        1.1,  # Adjust horizontal position for right label
        0.5,  # Vertical position (middle of colorbar)
        "High",  # Right label text
        va="center",
        ha="left",
        transform=cbar.ax.transAxes,
    )
    return plt.gcf()



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

    def load_mlflow_model(self) -> mlflow.pyfunc.PyFuncModel:
        """Loads the MLflow model."""
        model_uri = f"runs:/{self.cfg.models['graduation'].run_id}/model"

        try:
            load_model_func = {
                "sklearn": mlflow.sklearn.load_model,
                "xgboost": mlflow.xgboost.load_model,
                "lightgbm": mlflow.lightgbm.load_model,
                "pyfunc": mlflow.pyfunc.load_model,  # Default
            }.get(self.args.model_type, mlflow.pyfunc.load_model)
            model = load_model_func(model_uri)
            logging.info(
                "MLflow '%s' model loaded from '%s'", self.args.model_type, model_uri
            )
            return model
        except Exception as e:
            logging.error("Error loading MLflow model: %s", e)
            raise  # Critical error; re-raise to halt execution

    def predict(
        self, model: mlflow.pyfunc.PyFuncModel, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Performs inference and adds predictions to the DataFrame."""
        try:
            model_feature_names = model.named_steps["column_selector"].get_params()[
                "cols"
            ]
        except AttributeError:
            model_feature_names = model.metadata.get_input_schema().input_names()

        df_serving = df[model_feature_names]

        df_predicted = df_serving.copy()
        df_predicted["predicted_label"] = model.predict(df_serving)
        try:
            df_predicted["predicted_prob"] = model.predict_proba(df_serving)[:, 1]
        except AttributeError:
            logging.error("Model does not have predict_proba method.  Skipping.")
        return df_predicted

    def write_data_to_delta(self, df: pd.DataFrame, table_name_suffix: str):
        """Writes a DataFrame to a Delta Lake table."""
        write_schema = f"{self.args.databricks_institution_name}_silver"
        table_path = f"{self.args.DB_workspace}.{write_schema}.{self.args.db_run_id}_{table_name_suffix}"

        try:
            dataio.to_delta_table(df, table_path, spark_session=self.spark_session)
            logging.info(
                "%s data written to: %s", table_name_suffix.capitalize(), table_path
            )
        except Exception as e:
            logging.error(
                "Error writing %s data to Delta Lake: %s", table_name_suffix, e
            )
            raise  # Critical, prevent further execution.

    @staticmethod
    def predict_proba(
        X: pd.DataFrame,
        model: mlflow.pyfunc.PyFuncModel,
        feature_names: Optional[List[str]] = None,
        pos_label: Optional[bool | str] = None,
    ) -> np.ndarray:
        """Predicts probabilities using the provided model.  Handles data prep."""

        if feature_names is None:
            feature_names = model.named_steps["column_selector"].get_params()["cols"]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(data=X, columns=feature_names)
        # else: # This check seems unnecessary and potentially incorrect.
        #     assert X.shape[1] == len(feature_names)  # Check *number* of columns.
        pred_probs = model.predict_proba(X)
        if pos_label is not None:
            return pred_probs[:, model.classes_.tolist().index(pos_label)]
        else:
            return pred_probs

    def parallel_explanations(
        self,
        model: mlflow.pyfunc.PyFuncModel,
        df_features: pd.DataFrame,
        explainer: shap.Explainer,
        model_feature_names: List[str],
        n_jobs: Optional[int] = -1,
    ) -> shap.Explanation:
        """
        Calculates SHAP explanations in parallel using joblib.

        Args:
            model: mlflow.pyfunc.PyFuncModel.
            df_features pd.DataFrame: The feature dataset to calculate SHAP values for.
            explainer (shap.Explainer): The SHAP explainer object.
            model_feature_names (List[str]): List of feature names corresponding to the columns in `df_features`.
            n_jobs (Optional[int]): The number of jobs to run in parallel. Defaults to -1 (use all available CPUs).

        Returns:
            shap.Explanation: The combined SHAP explanation object.
        """

        logging.info("Calculating SHAP values for %s records", len(df_features))

        chunks = np.array_split(df_features, len(df_features) // 4)

        results = Parallel(n_jobs=n_jobs)(
            delayed(lambda model, chunk, explainer: explainer(chunk))(
                model, chunk, explainer
            )
            for chunk in chunks
        )

        combined_values = np.concatenate([r.values for r in results], axis=0)
        combined_data = np.concatenate([r.data for r in results], axis=0)
        combined_explanation = shap.Explanation(
            values=combined_values,
            data=combined_data,
            feature_names=model_feature_names,
        )
        return combined_explanation

    def calculate_shap_values(
        self,
        model: mlflow.pyfunc.PyFuncModel,
        df_processed: pd.DataFrame,
        model_feature_names: list[str],
    ) -> pd.DataFrame | None:
        """Calculates SHAP values."""

        try:
            # --- SHAP Values Calculation ---
            # TODO: Consider saving the explainer during training.
            shap_ref_data_size = 200  # Consider getting from config.

            # experiment_id = self.cfg.models[
            #     "graduation"
            # ].experiment_id  # Consider refactoring this
            # df_train = modeling.evaluation.extract_training_data_from_model(
            #     experiment_id
            # )
            
            df_train = dataio.from_delta_table(
            self.args.training_table_path, spark_session=self.spark_session
        )
            train_mode = df_train.mode().iloc[0]  # Use .iloc[0] for single row
            df_ref = (
                df_train.sample(
                    n=min(shap_ref_data_size, df_train.shape[0]),
                    random_state=self.cfg.random_state,
                )
                .fillna(train_mode)
                .loc[:, model_feature_names]
            )

            explainer = shap.explainers.KernelExplainer(
                ft.partial(
                    self.predict_proba,  # Use the static method
                    model=model,
                    feature_names=model_feature_names,
                    pos_label=self.cfg.pos_label,
                ),
                df_ref,
                link="identity",
            )

            # shap_values = explainer(df_processed[model_feature_names])

            shap_values_explanation = self.parallel_explanations(
                model=model,
                df_features=df_processed[model_feature_names],
                explainer=explainer,
                model_feature_names=model_feature_names,
                n_jobs=-1,
            )

            return shap_values_explanation
        except Exception as e:
            logging.error("Error during SHAP value calculation: %s", e)
            raise

    def get_top_features_for_display(
        self, df_serving, unique_ids, df_predicted, shap_values, model_feature_names
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
            result = inference.select_top_features_for_display(
                df_serving,
                unique_ids,
                pred_probs,
                shap_values.values,
                n_features=inference_params["num_top_features"],
                features_table=features_table,
                needs_support_threshold_prob=inference_params["min_prob_pos_label"],
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
        unique_ids = df_processed[self.cfg.student_id_col]

        model = self.load_mlflow_model()
        model_feature_names = model.named_steps["column_selector"].get_params()["cols"]

        # --- Email notify users ---
        # Uncomment below once we want to enable CC'ing to DK's email.
        # Secrets from Databricks
        w = WorkspaceClient()
        MANDRILL_USERNAME = w.dbutils.secrets.get(scope="sst", key="MANDRILL_USERNAME")
        MANDRILL_PASSWORD = w.dbutils.secrets.get(scope="sst", key="MANDRILL_PASSWORD")
        SENDER_EMAIL = Address("Datakind Info", "help", "datakind.org")
        send_inference_kickoff_email(
            SENDER_EMAIL,
            [self.args.notification_email],
            [self.args.DK_CC_EMAIL],
            MANDRILL_USERNAME,
            MANDRILL_PASSWORD,
        )
        # send_inference_kickoff_email(
        #     SENDER_EMAIL, [notification_email], [], MANDRILL_USERNAME, MANDRILL_PASSWORD
        # )

        df_predicted = self.predict(model, df_processed)
        self.write_data_to_delta(df_predicted, "predicted_dataset")

        # --- SHAP Values Calculation ---
        shap_values = self.calculate_shap_values(
            model, df_processed, model_feature_names
        )

        if shap_values is not None:  # Proceed only if SHAP values were calculated
            # --- SHAP Summary Plot ---
            shap_fig = plot_shap_beeswarm(shap_values)

            shap_results = self.get_top_features_for_display(
                df_processed, unique_ids, df_predicted, shap_values, model_feature_names
            )
            # --- Save Results to ext/ folder in Gold volume. ---
            if shap_results is not None:
                # Specify the folder for the output files to be stored.
                result_path = f"{self.args.job_root_dir}/ext/"
                os.makedirs(result_path, exist_ok=True)
                print("result_path:", result_path)

                # Write the DataFrame to Unity Catalog table
                self.write_data_to_delta(shap_results, "shap_results_dataset")

                # Write the DataFrame to CSV in the specified volume
                spark_df = self.spark_session.createDataFrame(shap_results)
                spark_df.coalesce(1).write.format("csv").option("header", "true").mode(
                    "overwrite"
                ).save(result_path + "inference_output")
                # Write the SHAP chart png to the volume
                shap_fig.savefig(result_path + "shap_chart.png", bbox_inches="tight")
            else:
                logging.error(
                    "Empty Shap results, cannot create the SHAP chart and table"
                )
                raise Exception(
                    "Empty Shap results, cannot create the SHAP chart and table"
                )

        # --- Write Inference Dataset --- (This was missing, but good to have)
        self.write_data_to_delta(df_processed[model_feature_names], "inference_dataset")


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
        required=True,
        type=str,
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
        "--DK_CC_EMAIL",
        type=str,
        required=True,
        help="Datakind email address CC'd",
    )

    parser.add_argument(
        "--training_table_path",
        type=str,
        required=True,
        help="Path to training dataset table",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        sys.path.append(f"/Workspace/Users/pedro.melendez@datakind.org/python-refactor-pipeline/pipelines/tasks/utils/")
        schemas = importlib.import_module(f'{args.databricks_institution_name}.schemas')
        logging.info("Running task with custom schema")
    except:
        print("Running task with default schema")
        from student_success_tool.schemas import pdp as schemas
        logging.info("Running task with default schema")
    task = ModelInferenceTask(args)
    task.run()
