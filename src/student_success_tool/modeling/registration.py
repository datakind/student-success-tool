import logging
import typing as t

import mlflow
import mlflow.exceptions
import mlflow.tracking
import os
import shutil
import uuid
import pandas as pd
import numpy as np
from typing import List

LOGGER = logging.getLogger(__name__)


try:
    # Try to connect using Databricks Connect
    from databricks.connect import DatabricksSession

    spark = DatabricksSession.builder.getOrCreate()
    print("Using remote Spark session (Databricks Connect).")
except Exception as e:
    # Fallback to local Spark session
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.master("local[*]").appName("LocalTest").getOrCreate()
    print("Falling back to local Spark session:", str(e))


def register_mlflow_model(
    model_name: str,
    institution_id: str,
    *,
    run_id: str,
    catalog: str,
    registry_uri: str = "databricks-uc",
    model_alias: t.Optional[str] = "Staging",
    mlflow_client: mlflow.tracking.MlflowClient,
) -> None:
    """
    Register an mlflow model according to one of their various recommended approaches.

    Args:
        model_name
        institution_id
        run_id
        catalog
        registry_uri
        model_alias
        mlflow_client

    References:
        - https://mlflow.org/docs/latest/model-registry.html
    """
    model_path = f"{catalog}.{institution_id}_gold.{model_name}"
    mlflow.set_registry_uri(registry_uri)

    try:
        mlflow_client.create_registered_model(name=model_path)
        LOGGER.info("new registered model '%s' successfully created", model_path)
    except mlflow.exceptions.MlflowException as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            LOGGER.info("model '%s' already created in registry", model_path)
        else:
            raise e

    model_uri = get_mlflow_model_uri(run_id=run_id, model_path="model")
    mv = mlflow_client.create_model_version(model_path, source=model_uri, run_id=run_id)
    if model_alias is not None:
        mlflow_client.set_registered_model_alias(
            model_path, alias=model_alias, version=mv.version
        )
    LOGGER.info("model version successfully registered at '%s'", model_path)


def log_confusion_matrix(
    institution_id: str,
    *,
    run_id: str,
    catalog: str,
) -> None:
    """
    Register an mlflow model according to one of their various recommended approaches.

    Args:
        institution_id
        run_id
        catalog
    """
    confusion_matrix_table_path = (
        f"{catalog}.{institution_id}_gold.inference_{run_id}_confusion_matrix"
    )

    def safe_div(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    try:
        run = mlflow.get_run(run_id)
        required_metrics = [
            "test_true_positives",
            "test_true_negatives",
            "test_false_positives",
            "test_false_negatives",
        ]
        metrics = {m: run.data.metrics.get(m) for m in required_metrics}

        if any(v is None for v in metrics.values()):
            raise ValueError(
                f"Missing one or more required metrics in run {run_id}: {metrics}"
            )

        tp, tn, fp, fn = (
            metrics["test_true_positives"],
            metrics["test_true_negatives"],
            metrics["test_false_positives"],
            metrics["test_false_negatives"],
        )

        tn_percentage = safe_div(tn, tn + fp)
        tp_percentage = safe_div(tp, tp + fn)
        fp_percentage = safe_div(fp, fp + tn)
        fn_percentage = safe_div(fn, fn + tp)

        confusion_matrix_table = pd.DataFrame(
            {
                "true_positive": [tp_percentage],
                "false_positive": [fp_percentage],
                "true_negative": [tn_percentage],
                "false_negative": [fn_percentage],
            }
        )

        confusion_matrix_table_spark = spark.createDataFrame(confusion_matrix_table)
        confusion_matrix_table_spark.write.mode("overwrite").saveAsTable(
            confusion_matrix_table_path
        )
        LOGGER.info(
            "Confusion matrix written to table '%s' for run_id=%s",
            confusion_matrix_table_path,
            run_id,
        )

    except mlflow.exceptions.MlflowException as e:
        raise RuntimeError(f"MLflow error while retrieving run {run_id}: {e}")
    except Exception:
        LOGGER.exception("Failed to compute or store confusion matrix.")
        raise


def log_roc_table(
    institution_id: str,
    *,
    run_id: str,
    catalog: str,
) -> None:
    """
    Computes and saves an ROC curve table (FPR, TPR, threshold, etc.) for a given MLflow run
    by reloading the test dataset and the trained model.

    Args:
        institution_id (str): Institution ID prefix for table name.
        run_id (str): MLflow run ID of the trained model.
        catalog (str): Destination catalog/schema for the ROC curve table.
    """
    data_run_tag = "Training Data Storage and Analysis"
    table_path = f"{catalog}.{institution_id}_gold.sample_training_{run_id}_roc_curve"
    tmp_dir = f"/tmp/{uuid.uuid4()}"  # unique tmp path

    try:
        client = mlflow.tracking.MlflowClient()
        model_run = client.get_run(run_id)
        experiment_id = model_run.info.experiment_id

        # Find the run that logged the data
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id], output_format="pandas"
        )
        assert isinstance(runs_df, pd.DataFrame)

        data_run_id = runs_df[runs_df["tags.mlflow.runName"] == data_run_tag][
            "run_id"
        ].item()

        # Load test data
        artifact_path = mlflow.artifacts.download_artifacts(
            run_id=data_run_id, artifact_path="data", dst_path=tmp_dir
        )
        df = pd.read_parquet(os.path.join(artifact_path, "training_data"))
        test_df = df[df["_automl_split_col_0000"] == "test"].copy()

        # Load model + features
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        feature_names: List[str] = model.named_steps["column_selector"].get_params()[
            "cols"
        ]

        # Infer target column
        excluded_cols = set(feature_names + ["_automl_split_col_0000"])
        target_cols = list(set(test_df.columns) - excluded_cols)
        if len(target_cols) != 1:
            raise ValueError(f"Could not infer a single target column: {target_cols}")
        target_col = target_cols[0]

        # Prepare inputs for ROC
        y_true = test_df[target_col].values
        X_test = test_df[feature_names]
        y_scores = model.predict_proba(X_test)[:, 1]  # probabilities for class 1

        # Calculate ROC table manually and plot all thresholds.
        # Down the line, we might want to specify a threshold to reduce plot density
        thresholds = np.sort(np.unique(y_scores))[::-1]
        P, N = np.sum(y_true == 1), np.sum(y_true == 0)

        rows = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            TN = np.sum((y_pred == 0) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))
            TPR = TP / P if P else 0
            FPR = FP / N if N else 0
            rows.append(
                {
                    "threshold": round(thresh, 4),
                    "true_positive_rate": round(TPR, 4),
                    "false_positive_rate": round(FPR, 4),
                    "true_positive": int(TP),
                    "false_positives": int(FP),
                    "true_negatives": int(TN),
                    "false_negatives": int(FN),
                }
            )

        roc_df = pd.DataFrame(rows)
        spark_df = spark.createDataFrame(roc_df)
        spark_df.write.mode("overwrite").saveAsTable(table_path)

    except Exception as e:
        raise RuntimeError(f"Failed to log ROC table for run {run_id}: {e}") from e
    finally:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def get_model_name(
    *,
    institution_id: str,
    target: str,
    checkpoint: str,
    extra_info: t.Optional[str] = None,
) -> str:
    """
    Get a standard model name generated from key components, formatted as
    "{institution_id}_{target}_{checkpoint}[_{extra_info}]"
    """
    model_name = f"{institution_id}_{target}_{checkpoint}"
    if extra_info is not None:
        model_name = f"{model_name}_{extra_info}"
    return model_name


def get_mlflow_model_uri(
    *,
    model_name: t.Optional[str] = None,
    model_version: t.Optional[int] = None,
    model_alias: t.Optional[str] = None,
    run_id: t.Optional[str] = None,
    model_path: t.Optional[str] = None,
) -> str:
    """
    Get an mlflow model's URI based on its name, version, alias, path, and/or run id.

    References:
        - https://docs.databricks.com/gcp/en/mlflow/models
        - https://www.mlflow.org/docs/latest/concepts.html#artifact-locations
    """
    if run_id is not None and model_path is not None:
        return f"runs:/{run_id}/{model_path}"
    elif model_name is not None and model_version is not None:
        return f"models:/{model_name}/{model_version}"
    elif model_name is not None and model_alias is not None:
        return f"models:/{model_name}@{model_alias}"
    else:
        raise ValueError(
            "unable to determine model URI from inputs: "
            f"{model_name=}, {model_version=}, {model_alias=}, {model_path=}, {run_id=}"
        )
