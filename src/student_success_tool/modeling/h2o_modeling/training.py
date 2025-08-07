import logging
import typing as t

from mlflow.tracking import MlflowClient
import pandas as pd
from pandas.api.types import is_categorical_dtype, is_object_dtype, is_string_dtype

import h2o
from h2o.automl import H2OAutoML

from . import utils
from . import imputation

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
    Runs H2O AutoML for classification, with sklearn-based imputation (leakage-safe).
    """
    if client is None:
        client = MlflowClient()

    # Set and validate inputs
    seed = kwargs.pop("seed", 42)
    timeout_minutes = kwargs.pop("timeout_minutes", 5)
    exclude_cols = kwargs.pop("exclude_cols", [])
    split_col = kwargs.pop("split_col", "split")
    sample_weights_col = kwargs.pop("sample_weights_col", None)
    target_name = kwargs.pop("target_name")
    checkpoint_name = kwargs.pop("checkpoint_name")
    workspace_path = kwargs.pop("workspace_path")
    primary_metric = primary_metric.lower()

    required = {
        "target_name": target_name,
        "checkpoint_name": checkpoint_name,
        "workspace_path": workspace_path,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(f"Missing logging parameters: {', '.join(missing)}")

    if target_col not in df.columns:
        raise ValueError(f"Missing target_col '{target_col}' in DataFrame.")
    if split_col not in df.columns:
        raise ValueError(f"Missing split column '{split_col}' in DataFrame.")
    if primary_metric not in VALID_H2O_METRICS:
        raise ValueError(
            f"Invalid metric '{primary_metric}', must be one of {VALID_H2O_METRICS}."
        )

    if student_id_col and student_id_col not in exclude_cols:
        exclude_cols.append(student_id_col)

    # Set experiment
    experiment_id = utils.set_or_create_experiment(
        workspace_path, institution_id, target_name, checkpoint_name, client=client
    )

    # Fit and apply sklearn imputation
    LOGGER.info("Running sklearn-based imputation...")
    imputer = imputation.SklearnImputerWrapper()
    imputer.fit(df[df[split_col] == "train"])

    splits = ["train", "validate", "test"]
    df_splits = {
        split: imputer.transform(df[df[split_col] == split]) for split in splits
    }

    # Convert to H2OFrames and fix dtypes
    missing_flags = [col for col in df.columns if col.endswith("_missing_flag")]
    h2o_splits = {
        k: correct_h2o_dtypes(
            h2o.H2OFrame(v),
            v,
            force_enum_cols=missing_flags,
        )
        for k, v in df_splits.items()
    }
    for frame in h2o_splits.values():
        frame[target_col] = frame[target_col].asfactor()

    train, valid, test = h2o_splits["train"], h2o_splits["validate"], h2o_splits["test"]

    # Run H2O AutoML
    features = [
        col
        for col in df_splits["train"].columns
        if col not in exclude_cols + [target_col]
    ]
    LOGGER.info(f"Running H2O AutoML with {len(features)} features...")

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

    # Log experiment
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
        imputer=imputer,
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
