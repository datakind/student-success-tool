import logging
import typing as t

from mlflow.tracking import MlflowClient
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype

import h2o
from h2o.automl import H2OAutoML

from . import utils_h2o as utils
from . import imputation_h2o as imputation

LOGGER = logging.getLogger(__name__)

VALID_H2O_METRICS = {
    "auc",
    "logloss",
    "mean_per_class_error",
    "rmse",
    "mae",
    "aucpr",
}


def run_h2o_automl_classification(
    df: pd.DataFrame,
    *,
    target_col: str,
    primary_metric: str,
    institution_id: str,
    student_id_col: str,
    client: t.Optional["MLflowClient"] = None,
    **kwargs: object,
) -> tuple[str, H2OAutoML, h2o.H2OFrame, h2o.H2OFrame, h2o.H2OFrame]:
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

    if client is None:
        client = MlflowClient()

    # Set defaults and pop kwargs
    seed = kwargs.pop("seed", 42)
    timeout_minutes = kwargs.pop("timeout_minutes", 5)
    max_models = kwargs.pop("max_models", 100)
    exclude_cols = kwargs.pop("exclude_cols", [])
    split_col = kwargs.pop("split_col", "split")
    sample_weights_col = kwargs.pop("sample_weights_col", None)
    target_name = kwargs.pop("target_name", None)
    checkpoint_name = kwargs.pop("checkpoint_name", None)
    workspace_path = kwargs.pop("workspace_path", None)
    primary_metric = primary_metric.lower()

    # Validate inputs
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in input DataFrame.")

    if primary_metric not in VALID_H2O_METRICS:
        raise ValueError(
            f"Invalid primary_metric '{primary_metric}'. Must be one of {VALID_H2O_METRICS}"
        )

    missing_params = [
        name
        for name, value in {
            "target_name": target_name,
            "checkpoint_name": checkpoint_name,
            "workspace_path": workspace_path,
        }.items()
        if not value
    ]
    if missing_params:
        raise ValueError(f"Missing logging parameters: {', '.join(missing_params)}")

    if split_col not in df.columns:
        raise ValueError(
            "Input data must contain a 'split' column with values ['train', 'validate', 'test']."
        )

    if student_id_col and student_id_col not in exclude_cols:
        exclude_cols.append(student_id_col)

    # create experiment to log imputation
    experiment_id = utils.set_or_create_experiment(
        workspace_path,
        institution_id,
        target_name,
        checkpoint_name,
        client=client,
    )

    # Convert to H2OFrame and correct types
    # NOTE: H2O sometimes doesn't infer types correctly, so we need to manually check them here using our pandas DF.
    h2o_df = h2o.H2OFrame(df)
    h2o_df = correct_h2o_dtypes(h2o_df, df)

    # Convert dtype of target_col for h2o training, and create splits
    h2o_df[target_col] = h2o_df[target_col].asfactor()
    train = h2o_df[h2o_df[split_col] == "train"]
    valid = h2o_df[h2o_df[split_col] == "validate"]
    test = h2o_df[h2o_df[split_col] == "test"]

    # Define feature list
    features = [col for col in df.columns if col not in exclude_cols + [target_col]]

    LOGGER.info(
        f"Running H2O AutoML for target '{target_col}' with {len(features)} features..."
    )

    aml = H2OAutoML(
        max_runtime_secs=timeout_minutes * 60,
        sort_metric=primary_metric,
        stopping_metric=primary_metric,
        seed=seed,
        verbosity="info",
        include_algos=["XGBoost", "GBM", "GLM", "DRF"],
    )
    aml.train(
        x=features,
        y=target_col,
        training_frame=train,
        validation_frame=valid,
        leaderboard_frame=test,
        weights_column=sample_weights_col,
    )

    LOGGER.info(f"Best model: {aml.leader.model_id}")

    utils.log_h2o_experiment(
        aml=aml,
        train=train,
        valid=valid,
        test=test,
        institution_id=institution_id,
        target_col=target_col,
        target_name=target_name,
        checkpoint_name=checkpoint_name,
        workspace_path=workspace_path,
        experiment_id=experiment_id,
        client=client,
    )

    return experiment_id, aml, train, valid, test


def correct_h2o_dtypes(
    h2o_df: h2o.H2OFrame,
    original_df: pd.DataFrame,
    force_enum_cols: t.Optional[t.List[str]] = None,
    cardinality_threshold: int = 100,
) -> h2o.H2OFrame:
    """
    Correct H2OFrame dtypes based on original pandas DataFrame, targeting cases where
    originally non-numeric columns were inferred as numeric by H2O.

    Args:
        h2o_df: H2OFrame created from original_df
        original_df: Original pandas DataFrame with dtype info
        force_enum_cols: Optional list of column names to forcibly convert to enum
        cardinality_threshold: Max unique values to allow for enum conversion

    Returns:
        h2o_df (possibly modified)
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
            is_categorical_dtype(original_df[col])
            or is_object_dtype(original_df[col])
            or is_string_dtype(original_df[col])
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

            try:
                h2o_df[col] = h2o_df[col].asfactor()
                converted_columns.append(col)
            except Exception as e:
                LOGGER.warning(f"Failed to convert '{col}' to enum: {e}")

    LOGGER.info(
        f"H2O dtype correction complete. {len(converted_columns)} column(s) affected: {converted_columns}"
    )
    return h2o_df
