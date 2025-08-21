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
    client: t.Optional[MlflowClient] = None,
    **kwargs: object,
) -> tuple[str, H2OAutoML, h2o.H2OFrame, h2o.H2OFrame, h2o.H2OFrame]:
    if client is None:
        client = MlflowClient()

    # Set and validate inputs
    seed = kwargs.pop("seed", 42)
    timeout_minutes = int(float(str(kwargs.pop("timeout_minutes", 5))))
    exclude_cols = [
        c for c in t.cast(list[str], kwargs.pop("exclude_cols", [])) if c is not None
    ]
    split_col: str = str(kwargs.pop("split_col", "split"))
    sample_weight_col = str(kwargs.pop("sample_weight_col", "sample_weight"))

    target_name = kwargs.pop("target_name", None)
    checkpoint_name = kwargs.pop("checkpoint_name", None)
    workspace_path = kwargs.pop("workspace_path", None)
    metric = primary_metric.lower()

    if not all([target_name, checkpoint_name, workspace_path]):
        raise ValueError(
            "Missing logging parameters: target_name, checkpoint_name, workspace_path"
        )
    if target_col not in df or split_col not in df:
        raise ValueError("Missing target_col or split column in DataFrame.")
    if metric not in VALID_H2O_METRICS:
        raise ValueError(
            f"Invalid metric '{metric}', must be one of {VALID_H2O_METRICS}."
        )

    # Ensure columns that need to be excluded are from training & imputation
    if student_id_col and student_id_col not in exclude_cols:
        exclude_cols.append(student_id_col)

    must_exclude: set[str] = {target_col, split_col, sample_weight_col}
    for c in must_exclude:
        if c not in exclude_cols:
            exclude_cols.append(c)

    missing_cols = [c for c in exclude_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"exclude_cols contains missing columns: {missing_cols}")

    # Set training experiment
    experiment_id = utils.set_or_create_experiment(
        workspace_path=str(workspace_path),
        institution_id=institution_id,
        target_name=str(target_name),
        checkpoint_name=str(checkpoint_name),
        client=client,
    )

    # Fit and apply sklearn imputation
    LOGGER.info("Running sklearn-based imputation on feature columns only...")
    imputer = imputation.SklearnImputerWrapper()
    raw_model_features = [c for c in df.columns if c not in exclude_cols]
    imputer.fit(df.loc[df[split_col] == "train", raw_model_features])

    df_splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "validate", "test"):
        df_split = df[df[split_col] == split_name]
        X_transformed = imputer.transform(df_split[raw_model_features])

        LOGGER.info(
            f"X_transformed shape: {X_transformed.shape}, "
            f"exclude_cols shape: {df_split[exclude_cols].shape}, "
            f"original split shape: {df_split.shape}"
        )

        # Force imputer output to have the same index as the original split
        X_transformed.index = df_split.index

        # Combine transformed features with excluded columns
        df_split_processed = pd.concat([X_transformed, df_split[exclude_cols]], axis=1)

        df_splits[split_name] = df_split_processed

    # Convert to H2OFrames and fix dtypes
    h2o_splits: dict[str, h2o.H2OFrame] = {}
    for k, v in df_splits.items():
        missing_flags = [c for c in v.columns if c.endswith("_missing_flag")]
        hf = h2o.H2OFrame(v)
        hf = correct_h2o_dtypes(hf, v, force_enum_cols=missing_flags)
        hf[target_col] = hf[target_col].asfactor()
        h2o_splits[k] = hf

    train, valid, test = h2o_splits["train"], h2o_splits["validate"], h2o_splits["test"]

    # Run H2O AutoML
    processed_model_features = [c for c in train.columns if c not in exclude_cols]
    LOGGER.info(f"Running H2O AutoML with {len(processed_model_features)} features...")

    aml = H2OAutoML(
        max_runtime_secs=timeout_minutes * 60,
        sort_metric=metric,
        stopping_metric=metric,
        seed=seed,
        verbosity="info",
        include_algos=["XGBoost", "GBM", "GLM", "DRF"],
        nfolds=0,  # disable CV, use validation frame for early stopping
        # balance_classes=True,
    )
    aml.train(
        x=processed_model_features,
        y=target_col,
        training_frame=train,
        validation_frame=valid,
        leaderboard_frame=test,
        weights_column=sample_weight_col,
    )

    LOGGER.info(f"Best model: {aml.leader.model_id}")

    utils.log_h2o_experiment(
        aml=aml,
        train=train,
        valid=valid,
        test=test,
        target_col=target_col,
        experiment_id=experiment_id,
        imputer=imputer,
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
