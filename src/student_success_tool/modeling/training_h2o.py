import os
import datetime
import contextlib
import sys
import logging
import typing as t

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype

from . import evaluation_h2o as evaluation

import h2o
from h2o.automl import H2OAutoML

LOGGER = logging.getLogger(__name__)


def run_h2o_automl_classification(
    df: pd.DataFrame,
    *,
    target_col: str,
    primary_metric: str,
    institution_id: str,
    student_id_col: str,
    **kwargs: object,
) -> tuple[H2OAutoML, h2o.H2OFrame, h2o.H2OFrame, h2o.H2OFrame]:
    """
    Runs H2O AutoML for classification tasks and logs the best model to MLflow.

    Args:
        df: Input Pandas DataFrame with features and target.
        target_col: Name of the target column.
        primary_metric: Used to sort models; supports "logloss", "AUC", "AUCPR", etc.
        institution_id: Institution ID for experiment naming.
        student_id_col: Column name identifying students, excluded from training.
        **kwargs: Optional settings including timeout_minutes, max_models, etc.

    Returns:
        Trained H2OAutoML object.
    """

    # Validate input types
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(target_col, str):
        raise TypeError("target_col must be a string.")
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in input DataFrame.")

    VALID_H2O_METRICS = {"auc", "logloss", "mean_per_class_error", "rmse", "mae", "aucpr"}
    primary_metric = primary_metric.lower()
    if primary_metric not in VALID_H2O_METRICS:
        raise ValueError(f"Invalid primary_metric '{primary_metric}'. Must be one of {VALID_H2O_METRICS}")

    # Set defaults and pop kwargs
    seed = kwargs.pop("seed", 42)
    timeout_minutes = kwargs.pop("timeout_minutes", 5)
    max_models = kwargs.pop("max_models", 100)
    exclude_cols = kwargs.pop("exclude_cols", [])
    split_col = kwargs.pop("split_col", "split")

    if student_id_col and student_id_col not in exclude_cols:
        exclude_cols.append(student_id_col)

    # Convert to H2OFrame and correct types
    # NOTE: H2O sometimes doesn't infer types correctly, so we need to manually check them here using our pandas DF.
    h2o_df = h2o.H2OFrame(df)
    h2o_df = correct_h2o_dtypes(h2o_df, df)

    if split_col not in h2o_df.columns:
        raise ValueError("Input data must contain a 'split' column with values ['train', 'validate', 'test'].")

    h2o_df[target_col] = h2o_df[target_col].asfactor()
    train = h2o_df[h2o_df[split_col] == "train"]
    valid = h2o_df[h2o_df[split_col] == "validate"]
    test = h2o_df[h2o_df[split_col] == "test"]

    features = [col for col in df.columns if col not in exclude_cols + [target_col]]

    LOGGER.info(f"Running H2O AutoML for target '{target_col}' with {len(features)} features...")

    aml = H2OAutoML(
        max_runtime_secs=timeout_minutes * 60,
        max_models=max_models,
        sort_metric=primary_metric,
        seed=seed,
        verbosity="info",
        include_algos=["XGBoost", "GBM", "GLM"],
    )
    aml.train(x=features, y=target_col, training_frame=train, validation_frame=valid, leaderboard_frame=test)

    LOGGER.info(f"Best model: {aml.leader.model_id}")

    return aml, train, valid, test


def log_h2o_experiment(
    aml: H2OAutoML,
    *,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    institution_id: str,
    target_col: str,
    target_name: str,
    checkpoint_name: str,
    workspace_path: str,
    client: t.Optional[MLflowClient] = None,
):
    """
    Logs evaluation metrics, plots, and model artifacts for all models in an H2O AutoML leaderboard to MLflow.

    Args:
        aml: Trained H2OAutoML object.
        train: H2OFrame containing the training split.
        valid: H2OFrame containing the validation split.
        test: H2OFrame containing the test split.
        institution_id: Institution identifier, used to namespace the MLflow experiment.
        target_col: Column name of target (used for plotting and label extraction).
        target_name: Name of the target of the model from the config.
        checkpoint_name: Name of the checkpoint of the model from the config.
        workspace_path: Path prefix for experiment naming within MLflow.
        client: Optional MLflowClient instance. If not provided, one will be created.

    Returns:
        experiment_id (str): The MLflow experiment ID used for logging.
        results_df (pd.DataFrame): DataFrame with metrics and MLflow run IDs for all successfully logged models.
    """
    LOGGER.info("Logging experiment to MLflow with classification plots...")

    if client is None:
        client = MlflowClient()

    experiment_id = set_or_create_experiment(
        workspace_path,
        institution_id,
        target_name,
        checkpoint_name,
        client=client,
    )

    results = []
    leaderboard_df = aml.leaderboard.as_data_frame()
    top_model_ids = leaderboard_df["model_id"].tolist()

    if not top_model_ids:
        LOGGER.warning("No models found in leaderboard.")
        return experiment_id, pd.DataFrame()

    for idx, model_id in enumerate(top_model_ids):
        # Log every 10 models
        if idx % 10 == 0:
            LOGGER.info(f"Evaluating model {idx + 1}/{len(top_model_ids)}: {model_id}")

        # Setting threshold to 0.5 due to binary classification
        metrics = evaluate_and_log_model(aml, model_id, train, valid, test, 0.5, client)

        if metrics:
            results.append(metrics)

    results_df = pd.DataFrame(results)
    LOGGER.info(f"Logged {len(results_df)} model runs to MLflow.")

    return experiment_id, results_df


def evaluate_and_log_model(
    aml: H2OAutoML,
    model_id: str,
    train: H2OFrame,
    valid: H2OFrame,
    test: H2OFrame,
    client: MlflowClient,
    threshold: float = 0.5,
) -> dict | None:
    """
    Evaluates a single H2O model at a given threshold and logs metrics, plots, and artifacts to MLflow.

    Args:
        model_id (str): The H2O model ID to evaluate and log.
        aml: H2OAutoML object containing the leaderboard and trained models.
        train: H2OFrame with training data.
        valid: H2OFrame with validation data.
        test: H2OFrame with test data.
        threshold (float): Threshold to apply for binary classification metrics.
        client (MlflowClient): Initialized MLflow client used for logging.

    Returns:
        dict: Dictionary of evaluation metrics including `mlflow_run_id`, or `None` if evaluation fails.
    """
    try:
        model = h2o.get_model(model_id)

        with contextlib.redirect_stdout(sys.__stdout__), contextlib.redirect_stderr(sys.__stderr__):
            metrics = evaluation.get_metrics_near_threshold_all_splits(model, train, valid, test, threshold=threshold)

            with mlflow.start_run(run_name=f"h2o_eval_{model_id}"):
                run_id = mlflow.active_run().info.run_id

                for k, v in metrics.items():
                    if k != "model_id":
                        mlflow.log_metric(k, v)

                for split_name, frame in zip(["train", "val", "test"], [train, valid, test]):
                    y_true = frame["target"].as_data_frame().values.flatten()
                    preds = model.predict(frame)
                    positive_class_label = preds.col_names[-1]
                    y_proba = preds[positive_class_label].as_data_frame().values.flatten()
                    y_pred = (y_proba >= 0.5).astype(int)

                    evaluation.generate_all_classification_plots(y_true, y_pred, y_proba, prefix=split_name)

                local_model_dir = f"/tmp/h2o_models/{model_id}"
                os.makedirs(local_model_dir, exist_ok=True)
                h2o.save_model(model, path=local_model_dir, force=True)
                mlflow.log_artifacts(local_model_dir, artifact_path="model")

        metrics["mlflow_run_id"] = run_id
        return metrics

    except Exception as e:
        LOGGER.exception(f"Failed to log model {model_id}: {e}")
        return None


def set_or_create_experiment(
    workspace_path: str,
    institution_id: str,
    target_name: str,
    checkpoint_name: str,
    client: t.Optional[MlflowClient] = None,
) -> str:
    """
    Creates or retrieves a structured MLflow experiment and sets it as the active experiment.

    Args:
        workspace_path: Base MLflow workspace path.
        institution_id: Institution or tenant identifier used for experiment naming.
        target_name: Name of the target variable.
        checkpoint_name: Name of the modeling checkpoint.
        client: MLflow client. A new one is created if not provided.

    Returns:
        MLflow experiment ID (created or retrieved).
    """
    if client is None:
        client = MlflowClient()

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    name_parts = [
        institution_id,
        target_name,
        checkpoint_name,
        "h20_automl",
        timestamp
    ]
    experiment_name = "/".join([
        workspace_path.rstrip("/"),
        "h2o_automl",
        "_".join([part for part in name_parts if part]),
    ])

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        raise RuntimeError(f"Failed to create or set MLflow experiment: {e}")


def correct_h2o_dtypes(h2o_df, original_df, force_enum_cols=None, cardinality_threshold=100):
    """
    Correct H2OFrame dtypes based on original pandas DataFrame, targeting cases where
    originally non-numeric columns were inferred as numeric by H2O.
    
    Args:
        h2o_df: H2OFrame created from original_df
        original_df: Original pandas DataFrame with dtype info
        force_enum_cols: Optional list of column names to forcibly convert to enum
        cardinality_threshold: Max unique values to allow for enum conversion
    
    Returns:
        h2o_df (possibly modified), and a list of column names proposed or actually converted
    """
    force_enum_cols = set(force_enum_cols or [])
    converted_columns = []

    LOGGER.info("Starting H2O dtype correction.")

    for col in original_df.columns:
        if col not in h2o_df.columns:
            LOGGER.debug(f"Skipping '{col}': not found in H2OFrame.")
            continue

        orig_dtype = original_df[col].dtype
        h2o_type = h2o_df.types.get(col)
        is_non_numeric = (
            is_categorical_dtype(original_df[col]) or
            is_object_dtype(original_df[col]) or
            is_string_dtype(original_df[col])
        )
        h2o_is_numeric = h2o_type in ("int", "real")
        nunique = original_df[col].nunique(dropna=True)

        LOGGER.debug(
            f"Column '{col}': orig_dtype={orig_dtype}, h2o_dtype={h2o_type}, "
            f"non_numeric={is_non_numeric}, unique={nunique}"
        )

        should_force = col in force_enum_cols
        needs_correction = (is_non_numeric and h2o_is_numeric) or should_force

        if needs_correction:
            if not should_force and nunique > cardinality_threshold:
                LOGGER.warning(
                    f"Skipping '{col}': high cardinality ({nunique}) for enum conversion."
                )
                continue

            LOGGER.info(
                f"Proposing '{col}' to enum "
                f"(originally {orig_dtype}, inferred as {h2o_type})."
            )

            converted_columns.append(col)

    LOGGER.info(f"H2O dtype correction complete. {len(converted_columns)} column(s) affected.")
    return h2o_df
