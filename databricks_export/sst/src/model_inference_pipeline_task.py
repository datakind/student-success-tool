# Databricks notebook source
"""
This script performs model inference for the Student Success Tool (SST) pipeline.

It loads a pre-trained ML model from MLflow Model Registry, reads a processed 
dataset from Delta Lake, performs inference, and writes the predictions back to 
Delta Lake.  It's designed to run within a Databricks environment.

"""

import logging

import mlflow
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils

import student_success_tool.dataio as dataio


# Disable mlflow autologging (prevents conflicts in Databricks environments).
mlflow.autolog(disable=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Suppress py4j logging.

# Attempt to create a Spark session. Handles non-Databricks environments.
try:
    spark_session = DatabricksSession.builder.getOrCreate()
except Exception:
    logging.warning("Unable to create Spark session; are you in a Databricks runtime?")
    spark_session = None

# --- Input Parameters (from Databricks widgets) ---
DB_workspace = dbutils.widgets.get("DB_workspace")  # Databricks workspace identifier.


institution_id = dbutils.widgets.get("institution_id")
model_name = dbutils.widgets.get("model_name")
sst_job_id = dbutils.widgets.get("sst_job_id")
model_type = dbutils.widgets.get("model_type")
# TODO: Determine how to track and pass the model URI.

# --- Delta Lake Configuration ---
catalog = DB_workspace
# write_schema = dbutils.jobs.taskValues.get(taskKey="data_ingestion", key="write_schema")  # From data ingestion task.
read_schema = f"{institution_id}_silver"
write_schema = f"{institution_id}_silver"
model_schema = f"{institution_id}_gold"

model_uri = f"models:/{catalog}.{model_schema}.{model_name}/1"  # Construct model URI.


# --- Model Loading ---
def mlflow_load_model(model_uri: str, model_type: str):
    """Loads an MLflow model based on its type."""

    load_model_func = {  # Dictionary to map model types to loading functions.
        "sklearn": mlflow.sklearn.load_model,
        "xgboost": mlflow.xgboost.load_model,
        "lightgbm": mlflow.lightgbm.load_model,
        "pyfunc": mlflow.pyfunc.load_model,  # Default if type is not recognized
    }.get(
        model_type, mlflow.pyfunc.load_model
    )  # Default to pyfunc if model_type not found.

    model = load_model_func(model_uri)
    logging.info(f"MLflow '{model_type}' model loaded from '{model_uri}'")
    return model


# Load the specified model.
loaded_model = mlflow_load_model(model_uri, model_type)

# --- Data Loading, Inference, and Saving ---
if spark_session:
    # Load the dataset prepared for inference.
    df_serving_dataset = dataio.from_delta_table(
        f"{catalog}.{read_schema}.{sst_job_id}_processed_dataset",
        spark_session=spark_session,
    )

    # Ensure the input data matches the model's expected input schema.
    try:
        model_columns = loaded_model.named_steps["column_selector"].get_params()["cols"]
    except (
        AttributeError
    ):  # if the model doesn't have a column_selector, it is likely a pyfunc model
        model_columns = loaded_model.metadata.get_input_schema().input_names()
    df_serving_dataset = df_serving_dataset[model_columns]

    # Write the inference-ready dataset back to Delta Lake.
    inference_dataset_path = f"{catalog}.{write_schema}.{sst_job_id}_inference_dataset"
    dataio.to_delta_table(
        df_serving_dataset, inference_dataset_path, spark_session=spark_session
    )

    # Reload the inference dataset (important for schema consistency with the model input).
    df_serving_dataset = dataio.from_delta_table(
        inference_dataset_path, spark_session=spark_session
    )

    # Perform inference.
    df_serving_dataset["predicted_label"] = loaded_model.predict(df_serving_dataset)

    try:
        df_serving_dataset["predicted_prob"] = loaded_model.predict_proba(
            df_serving_dataset
        )[:, 1]
    except (
        AttributeError
    ):  # if the model doesn't have predict_proba, it is likely a pyfunc model
        logging.warning(
            "Model does not have predict_proba method. Skipping probability prediction."
        )
        pass

    # Write the dataset with predictions to Delta Lake.
    predicted_dataset_path = f"{catalog}.{write_schema}.{sst_job_id}_predicted_dataset"
    dataio.to_delta_table(
        df_serving_dataset, predicted_dataset_path, spark_session=spark_session
    )
    logging.info(f"Predictions saved to: {predicted_dataset_path}")

    # TODO Shap values

else:
    logging.warning(
        "Spark session not initialized. Skipping data processing and inference."
    )

# COMMAND ----------

# TODO there are model dependencies that need to be installed at runtime
# This was the error receieved (although it still worked)
"""
- mlflow (current: 2.20.0, required: mlflow==2.19.0)
To fix the mismatches, call `mlflow.pyfunc.get_model_dependencies(model_uri)` to fetch the model's environment and install dependencies using the resulting environment file.
"""